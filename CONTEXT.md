# Recall — Branchable, Replayable Agent Memory

> Context document for Claude Code. Read this before any implementation work.
> Hackathon: CockroachDB × AWS "Build with Agentic Memory". Deadline Aug 18 2026.

---

## 1. One-line pitch

Recall is an agent memory kernel that makes agent memory **branchable** (fork it, try something risky, commit or throw it away) and **replayable** (reconstruct exactly what the agent knew at any past instant and prove which memories drove which decision).

## 2. The problem

Agent memory today is a flat, append-only blob. Two consequences block production deployment:

1. **No safe speculation.** An agent that explores a bad path pollutes its own memory permanently. There is no rollback.
2. **No provenance.** When an agent does something wrong, nobody can answer "what did it believe at the time, and which stored fact caused this?" That makes autonomous agents unshippable in any regulated or high-stakes workflow.

Recall fixes both by treating memory as a versioned, auditable system of record rather than a cache.

## 3. Why CockroachDB is load-bearing (not decorative)

This matters for the "Agentic Memory Design" judging criterion. Every core feature maps to a CockroachDB capability that a Postgres-in-a-container setup could not provide as cleanly:

| Recall feature | CockroachDB capability |
|---|---|
| Replay memory state at time T | `AS OF SYSTEM TIME` (MVCC time-travel queries) |
| Concurrent agents forking/merging without corruption | Serializable isolation by default |
| Semantic recall scoped to a branch | `VECTOR` type + distributed vector index |
| Live memory-write feed to the UI | Changefeeds |
| Memory that survives a region loss | Multi-region survival goals |
| Agent-driven infra ops and audit pulls | ccloud CLI |

## 4. Core concepts (use this vocabulary everywhere in code)

- **Memory** — one atomic fact the agent knows. Has content, an embedding, a source, a confidence, and a lifecycle (`active` / `superseded` / `retracted`).
- **Branch** — a named lineage of memory. Every branch has a parent and a fork point timestamp. Reads from a branch see its own memories plus everything the parent had at the fork point.
- **Recall** — hybrid retrieval: vector similarity plus structured filters, always scoped to a branch.
- **Decision** — a recorded agent action, joined to the exact memories that were recalled to produce it.
- **Replay** — reconstruct branch state at an arbitrary past timestamp and re-run a decision against it.
- **Audit** — append-only log of every read, write, fork, and commit, with actor identity.

## 5. Architecture

```
Demo agent (Lambda)  ──►  Recall MCP Server  ──►  Memory Kernel (Python)  ──►  CockroachDB Cloud
      │                        ▲                        │                          (multi-region)
      │                        │                        ├─► Bedrock (Titan embeddings)
      │                        │                        └─► Bedrock (Claude, reasoning)
      │                   Claude Code /
      │                   Cursor / VS Code
      ▼
  Web UI (timeline scrubber + memory graph)  ◄── changefeed ──┘
                                             └─► S3 (branch snapshots, audit archive)
```

### Component responsibilities

- **Memory Kernel** — pure Python library. All CockroachDB access lives here. No other component talks SQL.
- **Recall MCP Server** — thin MCP wrapper over the kernel. This is what makes Recall usable from Claude Code itself. Tools: `remember`, `recall`, `branch`, `commit`, `discard`, `diff`, `rewind`, `explain_decision`.
- **Demo agent** — a DevOps incident-triage agent running on Lambda. Chosen because it makes the branching and replay features obviously valuable rather than academic.
- **Web UI** — the demo surface. Timeline scrubber, branch tree, and a decision inspector showing memory provenance.
- **CockroachDB Cloud MCP Server** (the managed one, separate from ours) — used during development so Claude Code can inspect the live schema, and shown in the video as part of the workflow.

## 6. Required-tool coverage

CockroachDB tools used (need 2, we use 4):
1. Distributed Vector Indexing — core recall path
2. Cloud Managed MCP Server — dev workflow and live introspection
3. ccloud CLI — cluster provisioning, backup config, audit log export, all scripted
4. Agent Skills Repo — consumed during build, and we publish a Recall skill back

AWS services used (need 1, we use 5):
- Bedrock (Titan embeddings + Claude for agent reasoning)
- Lambda (agent execution)
- S3 (branch snapshot export, audit archive)
- API Gateway (UI to kernel)
- CloudWatch (observability)

## 7. Data model (target shape, refine in Phase 1)

- `branches` — id, name, parent_branch_id, fork_point_ts, status, created_by, created_at
- `memories` — id, branch_id, kind, content, embedding VECTOR, source, confidence, status, superseded_by, created_at
- `decisions` — id, branch_id, agent_id, input_hash, action, rationale, outcome, created_at
- `decision_memories` — decision_id, memory_id, similarity, rank
- `audit_log` — id, actor, op, target_type, target_id, payload, ts

Vector index on `memories.embedding`. Branch-scoped reads resolve through a recursive CTE over the branch ancestry plus `AS OF SYSTEM TIME` at each fork point.

## 8. Build phases (20–23 days)

| Phase | Days | Output |
|---|---|---|
| 0. Scaffold and provision | 1–2 | Repo, license, ccloud-provisioned cluster, env config, CI |
| 1. Schema and kernel core | 3–5 | Migrations, write/read paths, audit log, tests |
| 2. Embeddings and recall | 6–8 | Bedrock embeddings, vector index, hybrid branch-scoped recall |
| 3. Branching engine | 9–11 | fork / commit / discard / diff, ancestry resolution, concurrency tests |
| 4. Replay and provenance | 12–13 | AS OF SYSTEM TIME rewind, decision-to-memory provenance, explain_decision |
| 5. MCP server | 14–15 | Recall MCP server, works in Claude Code, read-only mode, auth |
| 6. Demo agent and UI | 16–19 | Lambda incident-triage agent, web UI with timeline and branch tree |
| 7. AWS deploy and hardening | 20–21 | IaC, S3 export, CloudWatch, RBAC, failure handling, load test |
| 8. Submission | 22–23 | README, arch diagram, sub-3-min video, Devpost writeup |

## 9. Non-negotiables

- Public repo, MIT license, detectable in the GitHub About section.
- Everything in the video must be running live against the real cluster. No mockups.
- The kernel must never be bypassed. If a component writes SQL directly, that is a bug.
- Every kernel operation writes an audit row in the same transaction as the operation.
- Secrets never committed. `.env.example` only.

## 10. Tech choices

Python 3.12, `psycopg` v3, `pgvector`-compatible typing where useful, `boto3` for Bedrock/S3, FastAPI for the HTTP layer, official Python MCP SDK, pytest, React + Vite + Tailwind for the UI, Terraform for AWS infra.

# Recall — Branchable, Replayable Agent Memory

Recall is an agent memory kernel that makes agent memory **branchable** (fork it,
try something risky, commit or throw it away) and **replayable** (reconstruct
exactly what the agent knew at any past instant and prove which memories drove
which decision).

Built for the CockroachDB × AWS "Build with Agentic Memory" hackathon. The full
project context and design rationale live in [`CONTEXT.md`](./CONTEXT.md).

## Why it matters

Agent memory today is a flat, append-only blob, which blocks two things
production agents need:

- **Safe speculation.** An agent that explores a bad path pollutes its memory
  permanently. Recall lets it fork a branch, try something, and discard it.
- **Provenance.** When an agent acts, Recall can answer "what did it believe at
  the time, and which stored fact caused this?" — every decision is joined to the
  exact memories that produced it.

Recall treats memory as a versioned, auditable system of record rather than a
cache.

## Why CockroachDB is load-bearing

| Recall feature | CockroachDB capability |
|---|---|
| Replay memory state at time T | `AS OF SYSTEM TIME` (MVCC time-travel) |
| Concurrent agents forking without corruption | Serializable isolation by default |
| Semantic recall scoped to a branch | `VECTOR` type + distributed vector index |
| Live memory-write feed to the UI | Changefeeds |
| Survive a region loss | Multi-region survival goals |
| Scripted infra ops and audit pulls | ccloud CLI |

## Repository layout

```
recall/
  kernel/       Memory kernel — the ONLY component that talks SQL to CockroachDB
  mcp_server/   Recall MCP server (thin wrapper over the kernel) — Phase 5
  agent/        Demo DevOps incident-triage agent (Lambda) — Phase 6
  ui/           Web UI: timeline scrubber + branch tree — Phase 6
  infra/        ccloud provisioning + audit export scripts
  migrations/   SQL migration files — Phase 1
  tests/        Pytest suite
```

Architectural rule: **the kernel is never bypassed.** If any component outside
`kernel/` writes SQL directly, that is a bug.

## Setup

Requires **Python 3.12+**.

```bash
# 1. Create a virtualenv and install (with dev tools)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# edit .env and fill in your real CRDB_CONNECTION_STRING and AWS settings
```

`.env` is git-ignored. Never commit real secrets — only `.env.example` with
placeholders is checked in.

### Environment variables

| Variable | Purpose |
|---|---|
| `CRDB_CONNECTION_STRING` | psycopg connection string for the cluster |
| `AWS_REGION` | AWS region for Bedrock / S3 |
| `BEDROCK_EMBEDDING_MODEL` | Bedrock model id for embeddings |
| `BEDROCK_REASONING_MODEL` | Bedrock model id for agent reasoning |
| `RECALL_ACTOR_ID` | Identity stamped on every audit-log row |
| `RECALL_READ_ONLY` | When true, kernel permits reads only |

## Provisioning the cluster

```bash
ccloud auth login --org <your-org>
./infra/provision.sh          # idempotent; exits 0 if the cluster exists
./infra/audit_export.sh       # pull audit logs as JSON (S3 upload lands Phase 7)
```

Some ccloud subcommand flags (service-account, api-key, backup config) are marked
with `TODO(verify)` in `infra/provision.sh` pending confirmation against the live
CLI — see the comments in that script.

## Development

```bash
ruff check .    # lint
pytest          # tests
```

CI runs ruff + pytest on every push and pull request
(see `.github/workflows/ci.yml`).

## Status

**Phase 0 — Scaffolding.** Repo structure, config, connection/retry plumbing, CI,
and infra scripts are in place. No schema, memory model, or Bedrock code yet;
those arrive in Phase 1 and beyond (see `CONTEXT.md` §8).

## License

This project is licensed under the [MIT License](./LICENSE).

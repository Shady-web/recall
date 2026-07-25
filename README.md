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

## Database schema & migrations

The schema (`branches`, `memories`, `decisions`, `decision_memories`,
`audit_log`) lives in `migrations/` as numbered SQL files. Apply them with the
migration runner, which is idempotent and safe to re-run:

```bash
python -m kernel.migrate                 # uses CRDB_CONNECTION_STRING
python -m kernel.migrate --dsn <dsn>     # explicit target
```

`memories.embedding` is `VECTOR(1024)` (matching Amazon Titan Text Embeddings V2).
Migration `002` adds the vector index on it. The runner seeds a root branch named
`main`.

**Vector index prerequisite.** Migration `002` creates a vector index, which
requires the cluster setting `feature.vector_index.enabled = true`. This is a
cluster-wide setting a database-scoped role may not be able to change, so the
migration does *not* set it; enable it once against your cluster before running
migrations:

```sql
SET CLUSTER SETTING feature.vector_index.enabled = true;
```

(Provisioning and the test harness handle this for you.)

## Embeddings & recall

Memories are embedded on write via `kernel/embeddings.py`:

- `BedrockEmbeddingProvider` calls Amazon Titan Text Embeddings V2
  (`amazon.titan-embed-text-v2:0`) through Bedrock, with retry/backoff on
  throttling.
- `FakeEmbeddingProvider` is a deterministic, offline provider so the whole test
  suite and the benchmark run with no AWS credentials or cost.

`kernel/recall.py` provides hybrid retrieval — vector similarity plus structured
SQL filters in one query (branch, `kind`, `min_confidence`, `since`, `status`),
returning a similarity score and rank per hit. The vector index is L2-only;
because embeddings are unit-normalized, L2 ranking equals cosine ranking and
similarity is reported as `1 - dist²/2`. Recall is branch-scoped today; ancestry
resolution across parent branches is a marked TODO seam for Phase 3.

Backfill embeddings for any pre-Phase-2 rows:

```bash
python -m kernel.backfill            # embeds memories with a NULL embedding
```

## Branching

Memory is branchable: fork a branch, speculate on it, then commit or discard.

```python
fork(db, actor, "main", "hotfix")     # child branch, fork_point_ts = now()
kernel.remember("hotfix", ...)        # never affects the parent
kernel.recall("hotfix", "...")        # sees own memories + parent as of the fork
commit(db, actor, "hotfix")           # replays onto parent, or returns conflicts
discard(db, actor, "hotfix")          # marks discarded; nothing is deleted
diff(db, actor, "main", "hotfix")     # added / superseded / retracted per side
```

Run `python scripts/demo_branching.py` for an end-to-end walkthrough.

### Ancestry uses logical time-travel, not `AS OF SYSTEM TIME`

`CONTEXT.md` §7 originally assumed a recursive CTE with `AS OF SYSTEM TIME` at
each fork point. **That is not implementable on CockroachDB**, verified against
v25.2:

- AOST must be attached to a **top-level statement**. In a CTE, sub-select, or
  one arm of a `UNION` it fails with `AS OF SYSTEM TIME must be provided on a
  top-level statement` — so one statement cannot read different ancestry
  segments at different timestamps.
- AOST cannot look back past MVCC garbage collection (`gc.ttlseconds`, default
  4h), so a branch outliving the GC window would become unreadable.

Ancestry therefore bounds each segment with `created_at <= fork_point`, and
status changes carry timestamps (`superseded_at` / `retracted_at`, migration
003) so "status as of T" is computable in SQL. This is exact for append-only
memory rows, is not GC-bounded, and keeps the read in one index-accelerated
statement.

`AS OF SYSTEM TIME` remains load-bearing for **Phase 4 replay**, where a single
whole-statement timestamp is the right tool — verified working, and verified
compatible with the vector index.

One structural constraint worth knowing: writing
`branch_id IN (SELECT id FROM ancestry_cte)` **defeats the vector index** (the
planner full-scans). So `resolve_ancestry()` runs the recursive CTE over the tiny
`branches` table first and the resolved ids are passed as a literal list, which
preserves index prefix spans. A regression test asserts this.

## Replay & decision provenance

Recall answers two different "what did we know?" questions, and **they can
disagree — that difference is the point**:

| | `replay_branch_at(branch, t)` | `replay_cluster_at(t)` |
|---|---|---|
| Question | what did this branch **logically contain**? | what did the cluster **physically look like**? |
| Mechanism | validity columns (`created_at` / `superseded_at` / `retracted_at`) + ancestry | `SET TRANSACTION AS OF SYSTEM TIME` |
| Age limit | **none** — works at any age | GC-bounded (`gc.ttlseconds`, 4h default) |
| Use | durable replay, provenance, re-runs | forensic: true historical bytes, incl. in-place edits |

`replay_window_bounds()` reports the currently safe physical-replay range, and
`replay_cluster_at()` raises a typed `ReplayWindowExpiredError` naming that window
rather than returning misleading data. (The raw engine error past the window is
actively unhelpful — e.g. `database ... does not exist`.)

```python
explain_decision(db, actor, decision_id)   # what drove it + what's since invalid
rewind_and_rerun(db, actor, decision_id, agent)              # faithful replay
rewind_and_rerun(db, actor, decision_id, agent, as_of=now)   # decide again today
```

`explain_decision` returns each contributing memory's similarity and rank **as
recorded at decision time** alongside its **current** status, and flags — at the
top level — any decision resting on memory since superseded or retracted.

`rewind_and_rerun` takes an **injected** agent callable, so `replay.py` stays free
of agent/Bedrock specifics. Run `python scripts/demo_replay.py` for the full
walkthrough.

## Benchmark

`benchmarks/bench_recall.py` loads 50k synthetic memories and reports recall
latency (p50/p95/p99). It uses the fake provider, so it needs no AWS. Committed
results live in [`benchmarks/recall_benchmark.md`](./benchmarks/recall_benchmark.md).

```bash
python benchmarks/bench_recall.py --count 50000 --queries 300
```

## Development

```bash
ruff check .    # lint
pytest          # tests
```

CI runs ruff + pytest on every push and pull request
(see `.github/workflows/ci.yml`).

### Running the tests

The kernel tests run against a **live CockroachDB instance** (v25.2+, for the
`VECTOR` type). A local insecure single-node is fine:

```bash
docker run -d --name recall-crdb -p 26257:26257 \
    cockroachdb/cockroach:latest-v25.2 start-single-node --insecure

# point the tests at it (this is the default if unset)
export RECALL_TEST_DSN="postgresql://root@localhost:26257/defaultdb?sslmode=disable"
pytest
```

Each test creates its own fresh, migrated database and drops it on teardown, so
runs are isolated. If no cluster is reachable at `RECALL_TEST_DSN`, the
database-backed tests are skipped (with a message) rather than failing.

## Status

**Phase 4 — Replay & provenance.** Logical replay (any age) and physical
`AS OF SYSTEM TIME` replay (GC-bounded, with a typed window guard),
`explain_decision` with prominent invalidated-memory flags, and
`rewind_and_rerun` against an injected agent. The MCP server arrives in Phase 5
(see `CONTEXT.md` §8).

## License

This project is licensed under the [MIT License](./LICENSE).

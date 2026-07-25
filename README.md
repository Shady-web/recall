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

**Phase 2 — Embeddings & recall.** Memories embed on write (Bedrock Titan V2,
with a deterministic fake provider for tests), the vector index is in place, and
hybrid branch-scoped recall combines vector similarity with structured SQL
filters. Recall is single-branch; branch forking, commit, and diff arrive in
Phase 3 (see `CONTEXT.md` §8).

## License

This project is licensed under the [MIT License](./LICENSE).

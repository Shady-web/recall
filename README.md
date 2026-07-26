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
  mcp_server/   Recall MCP server (thin wrapper over the kernel)
  agent/        Demo DevOps incident-triage agent (Lambda) — Phase 6
  ui/           Web UI: timeline scrubber + branch tree — Phase 6
  infra/        ccloud provisioning + audit export scripts
  migrations/   SQL migration files — Phase 1
  tests/        Pytest suite
```

Architectural rule: **the kernel is never bypassed.** If any component outside
`kernel/` writes SQL directly, that is a bug.

## Setup

Requires **Python 3.12+** and Docker (for the local test cluster).

```bash
# 1. Create a virtualenv and install (with dev tools)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start a local CockroachDB and run the tests
./scripts/dev_db.sh up
pytest

# 3. To talk to the real cloud cluster, configure the environment
cp .env.example .env
# edit .env and fill in your real CRDB_CONNECTION_STRING and AWS settings
```

`.env` is git-ignored. Never commit real secrets — only `.env.example` with
placeholders is checked in.

Local cluster for development and tests, CockroachDB Cloud for integration
checks and the demo — see **[DEV_SETUP.md](./DEV_SETUP.md)** for both.

### Environment variables

| Variable | Purpose |
|---|---|
| `CRDB_CONNECTION_STRING` | psycopg connection string for the cluster |
| `AWS_REGION` | AWS region for Bedrock / S3 |
| `BEDROCK_EMBEDDING_MODEL` | Bedrock model id for embeddings |
| `BEDROCK_REASONING_MODEL` | Bedrock model id for agent reasoning |
| `RECALL_ACTOR_ID` | Identity stamped on every audit-log row |
| `RECALL_READ_ONLY` | When true, kernel permits reads only |
| `RECALL_MCP_ACTOR` | MCP server identity (default: `RECALL_ACTOR_ID`); always prefixed `mcp:` |
| `RECALL_MCP_SERVER_NAME` | Name the MCP server advertises (default: `recall`) |

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
- AOST cannot look back past MVCC garbage collection. The reach is the cluster's
  `gc.ttlseconds` and varies by deployment (CockroachDB Cloud Basic reports
  4500s = 75 min; the self-hosted default is 14400s = 4h), so a branch outliving
  that window would become unreadable.

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
| Age limit | **none** — works at any age | GC-bounded (`gc.ttlseconds`, read live per-cluster) |
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

## MCP server

`mcp_server/` exposes the kernel over the Model Context Protocol (official Python
MCP SDK, stdio transport), so Recall is usable directly from Claude Code, Cursor,
and VS Code. This is **our** server — distinct from the
[managed CockroachDB Cloud MCP server](#cockroachdb-cloud-mcp-server-the-managed-one--read-only-by-design),
which we use during development to introspect the live cluster, and which is
read-only by design.

```bash
python -m mcp_server        # serve over stdio
```

### Tools

| Tool | Kind | Kernel entry point | Returns |
|---|---|---|---|
| `remember(branch, content, kind, source, confidence)` | write | `MemoryKernel.remember` | the stored memory |
| `recall(branch, query, k, filters)` | read | `MemoryKernel.recall` | hits with similarity + rank |
| `branch(parent, name)` | write | `branching.fork` | the new branch |
| `commit(branch)` | write | `branching.commit` | commit result, or structured conflicts |
| `discard(branch, reason)` | write | `branching.discard` | the discarded branch |
| `diff(branch_a, branch_b)` | read | `branching.diff` | added / superseded / retracted per side |
| `explain_decision(decision_id)` | read | `replay.explain_decision` | contributing memories + invalidation flags |
| `rewind(decision_id)` | read | `replay.rewind_summary` | logical replay summary + then-vs-now diff |

Each tool does exactly four things: validate its arguments, refuse the call if it
is a write on a read-only server, call **one** kernel entry point, serialize the
result. No SQL, no business logic, no state.

`rewind` is deliberately the *summary* path, not `rewind_and_rerun` — a tool call
must never fire an agent (and therefore a model call) as a side effect of what
the caller asked to be a read. To re-run an agent, call
`kernel.replay.rewind_and_rerun` directly.

### Everything is structured JSON

Two shapes, never prose and never a stack trace:

```json
{"ok": true,  "tool": "remember", "data": {"memory": {"id": "…", "kind": "fact"}}}
{"ok": false, "tool": "recall",   "error": {"type": "not_found",
                                            "message": "branch not found: nope",
                                            "retryable": false, "details": {}}}
```

`error.type` is a closed vocabulary — `read_only`, `not_found`, `invalid_state`,
`invalid_input`, `embedding_failed`, `replay_window_expired`, `internal` — so a
client can branch on it programmatically. Unexpected exceptions are logged with
their traceback *server-side only* and reported as `internal`.

**Commit conflicts are not errors.** A conflicting `commit` returns `ok: true`
with `committed: false`, `conflict_count > 0`, and a structured `conflicts` list;
the commit is a no-op and the branch stays open. A conflict is an outcome to
resolve, not a failure.

### Safety

- **Read-only mode.** With `RECALL_READ_ONLY=true`, the four write tools stay
  listed but return a typed `read_only` error, and their descriptions are
  prefixed so a model knows before calling. Hiding them would be simpler, but an
  agent that cannot see a tool concludes the capability does not exist and
  silently works around it; one that gets a clear refusal reports the real
  reason. The refusal is enforced at the MCP boundary *and* again in the kernel.
- **Audit identity.** Tools call the same kernel functions a direct Python caller
  would, so every operation still writes its audit row in the same transaction as
  the operation. What MCP adds is identity: the kernel is constructed with an
  actor carrying an `mcp:` prefix, so
  `SELECT * FROM audit_log WHERE actor LIKE 'mcp:%'` reliably separates
  MCP-initiated operations from scripts and the demo agent.
- **Boundary validation.** Every argument is validated before any kernel call, so
  a bad request never reaches the database and never buys an embedding. Unknown
  keys are rejected rather than ignored — a typo'd `min_confidance` that was
  silently dropped would return *more* memories than asked for while looking like
  it worked.

### Adding the server to your editor

The server needs `CRDB_CONNECTION_STRING` in its environment. It will read the
repo's `.env`, but only if the client happens to spawn it with the repo as the
working directory — so **pass the connection string explicitly** in the client
config, as below. Use absolute paths for the interpreter.

**Claude Code** — `.mcp.json` at the project root (checked in, shared with the
team):

```json
{
  "mcpServers": {
    "recall": {
      "command": "/absolute/path/to/recall/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "CRDB_CONNECTION_STRING": "postgresql://user:pass@cluster.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&sslrootcert=/absolute/path/root.crt",
        "AWS_REGION": "us-east-1",
        "RECALL_MCP_ACTOR": "claude-code",
        "RECALL_READ_ONLY": "false"
      }
    }
  }
}
```

Or from the CLI (note the `--` separating Claude's own flags from the command,
and that `--env` must not sit directly before the server name):

```bash
claude mcp add --env RECALL_MCP_ACTOR=claude-code --transport stdio recall \
  -- /absolute/path/to/recall/.venv/bin/python -m mcp_server
```

Verify with `/mcp` inside Claude Code; the eight tools appear as
`mcp__recall__remember`, `mcp__recall__recall`, and so on.

**Cursor** — `.cursor/mcp.json` in the project (or `~/.cursor/mcp.json` for all
projects). Same shape as Claude Code:

```json
{
  "mcpServers": {
    "recall": {
      "command": "/absolute/path/to/recall/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "CRDB_CONNECTION_STRING": "postgresql://user:pass@cluster.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&sslrootcert=/absolute/path/root.crt",
        "AWS_REGION": "us-east-1",
        "RECALL_MCP_ACTOR": "cursor"
      }
    }
  }
}
```

**VS Code** — `.vscode/mcp.json`. Note the different top-level key (`servers`,
not `mcpServers`) and the explicit `type`:

```json
{
  "servers": {
    "recall": {
      "type": "stdio",
      "command": "/absolute/path/to/recall/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {
        "CRDB_CONNECTION_STRING": "postgresql://user:pass@cluster.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&sslrootcert=/absolute/path/root.crt",
        "AWS_REGION": "us-east-1",
        "RECALL_MCP_ACTOR": "vscode"
      }
    }
  }
}
```

To give an editor read-only access to shared memory — useful for a reviewer, or
for any session you do not want writing to the team's `main` branch — set
`"RECALL_READ_ONLY": "true"` in that client's `env` block.

**Do not commit a config containing a real connection string.** Keep the
populated file local, or reference a variable your shell already exports.

## CockroachDB Cloud MCP server (the managed one) — read-only, by design

We use the **managed CockroachDB Cloud MCP server** during development so Claude
Code can inspect the live cluster directly — schema and index definitions,
`EXPLAIN` output, zone configuration, table statistics. It is what let us verify
empirically (rather than assume) that `VECTOR(1024)` is indexable, that
`AS OF SYSTEM TIME` cannot be nested in a CTE, and that ancestry-scoped recall
still plans as a `vector search`.

Register it with:

```bash
claude mcp add cockroachdb-cloud https://cockroachlabs.cloud/mcp \
  --transport http \
  --header "mcp-cluster-id: <your-cluster-id>" \
  --scope user
```

then authenticate with `/mcp` inside Claude Code.

**It is used for reads only. Every write goes through `kernel/`.**

This is a deliberate access-control decision, not a convention we hope people
follow. Recall's core guarantee is that *every* state change is attributable:
each kernel operation writes an `audit_log` row **in the same transaction** as
the operation itself, so an operation cannot commit without its audit record.
A direct `INSERT`/`UPDATE`/`DELETE` issued through a general-purpose SQL tool
bypasses that path entirely — it would mutate the system of record while leaving
no actor, no operation, and no provenance behind. Worse, it would do so
*invisibly*: nothing downstream could distinguish an unaudited change from one
that never happened, which quietly falsifies replay and
`explain_decision` for every decision that touched the affected memory.

So the rule is structural rather than advisory:

| Path | Allowed operations | Audited |
|---|---|---|
| `kernel/` (the only SQL in the project) | read + write | yes — same transaction |
| Cloud MCP server | schema/plan/config **inspection** | n/a — reads only |
| Anything else talking SQL directly | none | — this is a bug |

Two properties make the boundary enforceable rather than aspirational:
`MemoryKernel` cannot be constructed without an `actor`, so no write path can run
unattributed; and `RECALL_READ_ONLY=true` makes every write path raise
`ReadOnlyError` before touching the database, which is the setting to use for any
inspection-oriented session or shared credential.

For cluster *operations* — provisioning, backup configuration, audit-log export —
the scripted path is `infra/` via the ccloud CLI, not ad-hoc MCP calls, so
infrastructure changes are reviewable in version control alongside everything
else.

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

CI runs ruff + pytest on every push and pull request against a **real
CockroachDB** started by `scripts/dev_db.sh` — the same path you use locally, so
a green CI run means the database-backed tests actually executed rather than
skipped (see `.github/workflows/ci.yml`).

### Running the tests

The kernel tests run against a **live CockroachDB instance** (for the `VECTOR`
type). Start a local one and run them — no environment variable needed, the
tests already default to it:

```bash
./scripts/dev_db.sh up      # local single-node in Docker
pytest                      # 119 passed, 2 skipped, ~3.5 min
```

Each test creates its own fresh, migrated database and drops it on teardown, so
runs are isolated.

Point `RECALL_TEST_DSN` at the cloud cluster instead when you specifically want
to prove something there — expect **hours**, not minutes, because each test
rebuilds the vector index over the network (~92s per test vs ~2s locally).

**A skip is not a pass.** When no cluster is reachable the database-backed tests
*skip* rather than fail, so a misconfigured connection makes the suite exit 0
having run almost nothing. Run `pytest -ra` and check the counts: anything other
than 119 passed / 2 skipped means the tests are not reaching a cluster.

Full details — local vs cloud, Bedrock integration tests, troubleshooting — are
in **[DEV_SETUP.md](./DEV_SETUP.md)**.

## Status

**Phase 5 — MCP server.** The kernel is exposed over MCP as eight tools, usable
from Claude Code, Cursor, and VS Code, with typed JSON results, read-only
enforcement at the boundary, and `mcp:`-tagged audit identity. The demo agent and
web UI arrive in Phase 6 (see `CONTEXT.md` §8).

## License

This project is licensed under the [MIT License](./LICENSE).

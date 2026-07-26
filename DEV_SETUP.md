# Development setup

Recall talks to CockroachDB and nothing else, so the only real setup decision is
**which cluster you point at**. There are two, and they are for different jobs:

| | Local (Docker) | CockroachDB Cloud |
|---|---|---|
| **Use it for** | the test suite, day-to-day development | integration checks, benchmarks, the demo |
| **Full suite** | **~3.5 minutes** | ~3 hours |
| **Needs the network** | no | yes |
| **Setup** | `./scripts/dev_db.sh up` | `.env` + certificate |
| **Data** | throwaway, wiped on `down` | the real system of record |

Use local by default. Reach for the cloud when you specifically need to prove
something about the deployed cluster.

---

## 1. Python environment

Requires **Python 3.12+**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Local CockroachDB (the default for tests)

Requires Docker. Nothing else — no `ccloud`, no certificates, no secrets.

```bash
./scripts/dev_db.sh up      # start and wait until it accepts SQL
pytest                      # 119 passed, 2 skipped, ~3.5 min
```

That is the whole loop. The tests default to
`postgresql://root@localhost:26257/defaultdb?sslmode=disable`, which is exactly
what the script starts, so **no environment variable is needed**.

```bash
./scripts/dev_db.sh status  # running? which version?
./scripts/dev_db.sh sql     # interactive SQL shell
./scripts/dev_db.sh reset   # guaranteed-clean cluster (~3s)
./scripts/dev_db.sh down    # stop and discard
```

To run the kernel itself (migrations, demos) against local rather than the
cloud, override the connection string for that command only:

```bash
CRDB_CONNECTION_STRING="$(./scripts/dev_db.sh dsn)" python -m kernel.migrate
CRDB_CONNECTION_STRING="$(./scripts/dev_db.sh dsn)" python scripts/demo_branching.py
```

### Why local is 50x faster, and why that is not an accident

Every test creates its own database, migrates it, and drops it on teardown —
that is what keeps tests isolated. The cost is dominated by one statement:
building the vector index in migration `002`.

| | Local | Cloud |
|---|---|---|
| `CREATE DATABASE` | 0.01s | 4.0s |
| **migrate (vector index)** | **1.9s** | **83.4s** |
| `DROP DATABASE` | 0.25s | 4.6s |
| **per test** | **~2.1s** | **~92s** |

Across 121 tests that is 3.5 minutes versus over 3 hours. The isolation design
is right; running it over a network round-trip to another continent is what
makes it expensive.

### The image version is pinned on purpose

`scripts/dev_db.sh` pins `cockroachdb/cockroach:latest-v26.2` to match the cloud
cluster's major version (cloud runs v26.2.1, local v26.2.4). "Passes locally"
only means something if both speak the same dialect — Recall depends on the
`VECTOR` type, distributed vector indexing, and `AS OF SYSTEM TIME` semantics
that differ across releases. **Bump the pin when you bump the cloud cluster, not
independently.**

The container runs `--insecure` and binds to localhost. That is fine for
throwaway test databases and keeps setup to one command — but never point
`RECALL_TEST_DSN` at anything real while running insecure.

## 3. CockroachDB Cloud (integration checks and the demo)

Copy the example environment file and fill in real values:

```bash
cp .env.example .env
```

`.env` is git-ignored. Never commit real secrets — only `.env.example` with
placeholders is checked in.

You need `CRDB_CONNECTION_STRING` (from
`ccloud cluster connection-string <cluster> --sql-user <user> -o json`) and the
CA certificate it references, which `ccloud` writes to
`~/.postgresql/root.crt`.

Point the *test suite* at the cloud only when you actually want to prove
something against it:

```bash
export RECALL_TEST_DSN="$CRDB_CONNECTION_STRING"
pytest                      # slow; expect hours, see the table above
```

### Enable the vector index setting once

Migration `002` creates a vector index, which requires a cluster-wide setting.
The migration deliberately does not set it, because a database-scoped role may
not have permission to:

```sql
SET CLUSTER SETTING feature.vector_index.enabled = true;
```

`./scripts/dev_db.sh up` does this for you locally. On the cloud cluster it only
has to be done once, out of band.

### Run it on a stable connection

Long cloud runs are fragile on an unreliable link. In practice a full suite run
here took three attempts, twice interrupted by the local resolver dropping
(`failed to resolve host ... nodename nor servname provided`) roughly every 70
minutes. When that happens, resume rather than restart:

```bash
pytest --lf                 # re-run only what failed or errored
```

Distinguishing a real failure from a dropped connection matters: a network drop
shows as `psycopg.OperationalError` / `PoolTimeout` at *fixture setup*, with no
assertion in the traceback.

## 4. Bedrock integration tests

Two tests call Amazon Bedrock for real and are **off by default** — they are the
only tests in the suite that cost money and need AWS credentials:

```bash
RECALL_RUN_BEDROCK_INTEGRATION=1 pytest tests/test_integration_bedrock.py
```

Everything else uses `FakeEmbeddingProvider`, a deterministic offline provider,
so the suite needs no AWS access at all. A clean run reports
**119 passed, 2 skipped** — those two skips are these tests, and they are the
only legitimate skips in the suite.

## 5. Recall MCP server

The MCP server talks to whichever cluster its environment points at. For local
development:

```bash
CRDB_CONNECTION_STRING="$(./scripts/dev_db.sh dsn)" python -m mcp_server
```

Editor configuration for Claude Code, Cursor, and VS Code is in the
[README](./README.md#adding-the-server-to-your-editor).

## 6. Checks before pushing

```bash
ruff check .
pytest                      # against local; ~3.5 min
```

CI runs both on every push and pull request, **against a real CockroachDB**. The
workflow starts one with the same `./scripts/dev_db.sh up` you use locally, so a
green CI run means the database-backed tests actually executed.

### A skip is not a pass

The database-backed tests *skip* rather than fail when no cluster is reachable,
so a misconfigured connection makes the suite exit 0 having run almost nothing.
This has bitten this project already: `tests/conftest.py` probed the cluster with
a 3-second connect timeout, but the cloud cluster needs ~6 seconds, so the entire
suite reported green while executing none of it. The probe timeout is now 15
seconds — but timeouts are a fix for one cause, not for the class of problem.

Two guards now make a silent skip impossible in CI:

1. **`RECALL_REQUIRE_CRDB=1`** turns "no cluster reachable" from a skip into a
   hard collection error. Set it whenever a run is meant to prove something:

   ```bash
   RECALL_REQUIRE_CRDB=1 pytest      # exits 4 immediately if there is no cluster
   ```

2. **`scripts/check_ci_results.py`** asserts the observed counts from pytest's
   JUnit XML, so the run also fails if the suite silently shrinks:

   ```bash
   pytest -q --junitxml=pytest-report.xml
   python scripts/check_ci_results.py pytest-report.xml --min-passed 119 --max-skipped 2
   ```

**When you add tests, bump `--min-passed` in `.github/workflows/ci.yml`.** That
is deliberate friction: the number is a claim about how much of the suite really
runs, and it should only ever move up on purpose.

Locally, when a run looks suspiciously fast, check the counts rather than the
exit code:

```bash
pytest -ra                  # -ra reports every skip with its reason
```

Expect **119 passed, 2 skipped**. Anything else — especially a large skip count —
means the tests are not reaching a cluster.

### Why CockroachDB is a step, not a `services:` container

GitHub Actions service containers cannot be given a command, and this image's
entrypoint exits immediately without one:

```
/cockroach/cockroach.sh: line 278: error: mode unset, can be shell, bash,
or cockroach command (start-single-node, sql, etc.)
```

There is no `command:` key in the `services:` schema, so a service container
would fail its health check and never serve SQL. Running `./scripts/dev_db.sh up`
as a step avoids that, and has the better property anyway: CI and local
development share one code path, including the `feature.vector_index.enabled`
setting, so they cannot drift.

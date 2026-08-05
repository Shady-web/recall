#!/usr/bin/env bash
#
# run_demo.sh — bring up the whole Recall demo with one command.
#
# Seeds the incident corpus, starts the API bridge, starts the UI, waits until
# both are actually answering, and prints the URL.
#
# Usage:
#   ./scripts/run_demo.sh                 local cluster, fake embeddings, no cost
#   ./scripts/run_demo.sh --live          cloud cluster + real Bedrock (recording runs)
#   ./scripts/run_demo.sh --fresh         drop the local demo database first (local only)
#   ./scripts/run_demo.sh --reset         withdraw seeded rows in place, then re-seed
#   ./scripts/run_demo.sh --no-seed       leave existing data alone
#
# Use --fresh before a rehearsal or a recording. The demo story only lands from
# a clean start: once RB-014 has been withdrawn, a re-run recalls its
# replacement and decides correctly straight away, so the trap never springs.
# --reset withdraws the corpus without dropping anything (it works against the
# cloud cluster too) but leaves earlier incident branches in the tree.
#
# The default is deliberately the cheap path. Pointing at real Bedrock and the
# cloud cluster is an explicit --live, so nobody spends money by reflex.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_PORT="${RECALL_API_PORT:-8000}"
UI_PORT="${RECALL_UI_PORT:-5173}"
LOCAL_DSN="${RECALL_LOCAL_DSN:-postgresql://root@localhost:26257/recall_demo?sslmode=disable}"
ADMIN_DSN="${RECALL_ADMIN_DSN:-postgresql://root@localhost:26257/defaultdb?sslmode=disable}"
LOG_DIR="$ROOT/.demo-logs"

LIVE=0
RESET=0
FRESH=0
SEED=1

while [ $# -gt 0 ]; do
    case "$1" in
        --live)    LIVE=1 ;;
        --reset)   RESET=1 ;;
        --fresh)   FRESH=1 ;;
        --no-seed) SEED=0 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- colours, only when attached to a terminal ------------------------------
if [ -t 1 ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; RED=$'\033[31m'
    YEL=$'\033[33m'; CYA=$'\033[36m'; R=$'\033[0m'
else
    B=""; DIM=""; GRN=""; RED=""; YEL=""; CYA=""; R=""
fi

step() { printf '%s==>%s %s\n' "$CYA" "$R" "$1"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$R" "$1"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$R" "$1"; }
die()  { printf '%serror:%s %s\n' "$RED" "$R" "$1" >&2; exit 1; }

PIDS=()
cleanup() {
    printf '\n%s==>%s shutting down\n' "$CYA" "$R"
    for pid in "${PIDS[@]:-}"; do
        [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"

# --- preflight --------------------------------------------------------------
step "preflight"

[ -d .venv ] || die "no .venv found. See DEV_SETUP.md."
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv active ($(python3 --version))"

command -v node >/dev/null 2>&1 || die "node not found — the UI needs it."
ok "node $(node --version)"

if [ "$LIVE" -eq 1 ]; then
    [ -f .env ] || die "--live needs a .env with CRDB_CONNECTION_STRING and Bedrock auth."
    # --fresh drops a database. Refuse to point that at the cloud cluster: the
    # local demo database is disposable, the cloud one is not.
    [ "$FRESH" -eq 0 ] || die "--fresh is local-only; it will not drop a cloud database. Use --reset."
    export RECALL_OFFLINE=0
    unset RECALL_DSN
    SEED_FLAGS=""
    warn "LIVE mode: real Bedrock calls and the cloud cluster. This costs money."

    # The footgun this guards: the cheap path seeds with the FAKE embedding
    # provider, and --live then queries with Titan. Both emit 1024-dimension
    # unit vectors, so nothing about the shapes disagrees — but they are
    # different vector spaces, and every similarity score becomes orthogonal
    # noise. The kernel refuses this outright now; catching it here turns a
    # stack trace at API startup into an instruction.
    SPACES="$(python3 - <<'PY' 2>/dev/null || true
from kernel.config import settings
from kernel.db import Database, stored_embedding_spaces
db = Database(settings.crdb_connection_string)
print(",".join(sorted(s for s in stored_embedding_spaces(db) if s)))
db.close()
PY
)"
    case "$SPACES" in
        *fake*)
            die "this database holds FAKE embeddings ($SPACES), but --live queries
       with real Titan. Those vectors are not comparable — recall would return
       confident-looking hits scored on noise.
       Re-seed for live use:  ./scripts/run_demo.sh --live --reset"
            ;;
    esac
    [ -n "$SPACES" ] && ok "embedding space on record: $SPACES"
else
    export RECALL_OFFLINE=1
    export RECALL_DSN="$LOCAL_DSN"
    SEED_FLAGS="--offline --dsn $LOCAL_DSN"

    command -v docker >/dev/null 2>&1 || die "docker not found (needed for the local cluster)."
    if ! docker exec recall-crdb ./cockroach sql --insecure -e "SELECT 1" >/dev/null 2>&1; then
        warn "local cluster not up — starting it"
        ./scripts/dev_db.sh up >/dev/null || die "could not start the local cluster"
    fi
    ok "local cluster reachable"

    # The demo database and two cluster settings the kernel and the live feed
    # depend on: the vector index (recall) and rangefeeds (changefeeds).
    python3 - <<PY || die "could not prepare the demo database"
import psycopg
with psycopg.connect("$ADMIN_DSN", autocommit=True) as conn:
    conn.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
    conn.execute("SET CLUSTER SETTING kv.rangefeed.enabled = true")
    if $FRESH:
        conn.execute("DROP DATABASE IF EXISTS recall_demo CASCADE")
    conn.execute("CREATE DATABASE IF NOT EXISTS recall_demo")
PY
    if [ "$FRESH" -eq 1 ]; then
        ok "demo database dropped and recreated — the story starts from zero"
    else
        ok "vector index + rangefeeds enabled, recall_demo present"
    fi

    python3 -m kernel.migrate --dsn "$LOCAL_DSN" >/dev/null || die "migrations failed"
    ok "schema migrated"
fi

# --- seed -------------------------------------------------------------------
if [ "$SEED" -eq 1 ]; then
    step "seeding the incident corpus"
    if [ "$RESET" -eq 1 ]; then
        # shellcheck disable=SC2086
        python3 scripts/seed_incidents.py $SEED_FLAGS --reset >/dev/null 2>&1 || true
        ok "previous demo rows withdrawn"
    fi
    # shellcheck disable=SC2086
    python3 scripts/seed_incidents.py $SEED_FLAGS 2>&1 | grep -E '^seeded' || true
    ok "corpus on main"
else
    step "skipping seed (--no-seed)"
fi

# --- backend ----------------------------------------------------------------
step "starting the API bridge on :$API_PORT"
uvicorn api.main:app --port "$API_PORT" --host 127.0.0.1 \
    > "$LOG_DIR/api.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 40); do
    if curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then break; fi
    sleep 0.5
done
curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 \
    || die "API did not come up. Log: $LOG_DIR/api.log"

# The feed reports "starting" until the changefeed's first checkpoint arrives.
# Wait for it to settle so the mode printed below is the real one.
FEED_MODE="starting"
for _ in $(seq 20); do
    FEED_MODE="$(curl -s "http://127.0.0.1:$API_PORT/api/health" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["live_feed"]["mode"])')"
    [ "$FEED_MODE" != "starting" ] && break
    sleep 0.5
done

HEALTH="$(curl -s "http://127.0.0.1:$API_PORT/api/health")"
EMBED="$(printf '%s' "$HEALTH" | python3 -c 'import json,sys; print(json.load(sys.stdin)["embedding_provider"])')"
REASON="$(printf '%s' "$HEALTH" | python3 -c 'import json,sys; print(json.load(sys.stdin)["reasoning_provider"])')"
ok "API healthy — embeddings=$EMBED reasoning=$REASON"

if [ "$FEED_MODE" = "changefeed" ]; then
    ok "live feed: CockroachDB changefeed (push)"
else
    warn "live feed: $FEED_MODE — changefeed unavailable, the UI will say so"
fi

# The demo turns on RB-014 being *active* when the first incident fires. If a
# previous run already withdrew it, the agent recalls the replacement and
# decides correctly straight away — the trap never springs and step 2 has
# nothing to reveal. Say so at startup rather than letting it be discovered
# mid-demo.
STORY_READY="$(curl -s "http://127.0.0.1:$API_PORT/api/memories?branch=main&limit=200" \
    | python3 -c '
import json, sys
rows = json.load(sys.stdin)["memories"]
hit = [m for m in rows if m["metadata"].get("ref") == "RB-014"]
if not hit:
    print("missing")
else:
    print("ready" if any(m["status"] == "active" for m in hit) else "spent")
' 2>/dev/null || echo unknown)"

case "$STORY_READY" in
    ready) ok "RB-014 is active — the incident trap will spring" ;;
    spent)
        warn "RB-014 is ALREADY WITHDRAWN from an earlier run."
        warn "  The first incident will recall its replacement and decide correctly,"
        warn "  so the 'we were wrong' beat has nothing to reveal."
        warn "  Restart with:  ./scripts/run_demo.sh --fresh"
        ;;
    missing) warn "RB-014 not found on main — seed may not have run" ;;
    *)       warn "could not determine demo readiness" ;;
esac

# --- frontend ---------------------------------------------------------------
step "starting the UI on :$UI_PORT"
[ -d ui/node_modules ] || (cd ui && npm install >/dev/null 2>&1) \
    || die "npm install failed in ui/"

(cd ui && RECALL_API_URL="http://127.0.0.1:$API_PORT" RECALL_UI_PORT="$UI_PORT" \
    npm run dev > "$LOG_DIR/ui.log" 2>&1) &
PIDS+=($!)

UI_URL=""
for _ in $(seq 60); do
    if [ -f "$LOG_DIR/ui.log" ]; then
        UI_URL="$(grep -oE 'http://localhost:[0-9]+' "$LOG_DIR/ui.log" | head -1 || true)"
        [ -n "$UI_URL" ] && break
    fi
    sleep 0.5
done
[ -n "$UI_URL" ] || die "UI did not start. Log: $LOG_DIR/ui.log"
ok "UI serving"

# --- ready ------------------------------------------------------------------
cat <<EOF

  ${B}Recall is running.${R}

    ${B}${GRN}${UI_URL}${R}

  ${DIM}api      http://127.0.0.1:$API_PORT/api/health
  docs     http://127.0.0.1:$API_PORT/docs
  logs     $LOG_DIR/{api,ui}.log
  mode     $([ "$LIVE" -eq 1 ] && echo 'LIVE — real Bedrock + cloud cluster' || echo 'offline — fake embeddings, rule reasoner, local cluster')
  feed     $FEED_MODE${R}

  ${B}Demo path${R} ${DIM}(run these in order in a second terminal)${R}

  ${B}1. Fire the incident.${R} The agent recalls, forks, reasons, records provenance.
     It follows runbook RB-014 and decides to restart pgbouncer.
    ${DIM}curl -s localhost:$API_PORT/api/incidents -H 'content-type: application/json' \\
      -d '{"incident":"payments-svc checkout p99 latency 4.2s, pgbouncer cl_waiting rising"}'${R}

     Watch the branch appear in the tree without a reload — that is the
     changefeed. Open ${B}decision inspector${R}: provenance is clean, nothing flagged.

  ${B}2. Withdraw the bad runbook.${R} RB-014 turns out to have caused INC-2350.
    ${DIM}python3 scripts/seed_incidents.py $SEED_FLAGS --retract-runbook${R}

     The inspector header now badges the decision as resting on withdrawn
     memory, and RB-014's row goes red. Press ${B}rewind${R}: the faithful replay
     reproduces the original action, the re-run against today does not.

     In ${B}timeline${R}, drag back before the retraction: RB-014 is struck through
     (known then, gone now) and RB-031 is highlighted (learned since).

  ${B}3. Try to commit the bad branch.${R} It is refused as a no-op with a conflict —
     both the branch and main changed RB-014 after the fork. Conflicts come
     back as data, not as an error, and nothing is half-applied.

  ${B}4. Fire the incident again.${R} A fresh fork inherits the withdrawal, recalls
     RB-031 instead, decides correctly, and commits back to main cleanly.
    ${DIM}curl -s localhost:$API_PORT/api/incidents -H 'content-type: application/json' \\
      -d '{"incident":"payments-svc checkout p99 latency 4.2s again, cl_waiting climbing","commit":true}'${R}

  ${DIM}Ctrl-C to stop both.${R}

EOF

wait

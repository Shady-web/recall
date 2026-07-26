#!/usr/bin/env bash
#
# dev_db.sh — manage the local CockroachDB used by the test suite.
#
# The suite runs against a real cluster, but it does not have to be the *cloud*
# cluster. A local single-node instance is 40x faster per test (each test builds
# and drops its own migrated database, and the vector index dominates that cost:
# ~1.9s locally vs ~83s against CockroachDB Cloud) and it removes the network
# from the loop entirely. Keep the cloud cluster for integration checks and the
# demo; see DEV_SETUP.md.
#
# Usage:
#   ./scripts/dev_db.sh up       start the container and wait until it accepts SQL
#   ./scripts/dev_db.sh down     stop and remove it (all data is discarded)
#   ./scripts/dev_db.sh reset    down, then up — a guaranteed-clean cluster
#   ./scripts/dev_db.sh status   is it running, and what version
#   ./scripts/dev_db.sh dsn      print the DSN the tests default to
#   ./scripts/dev_db.sh sql      open an interactive SQL shell
#
# Requirements: Docker. Nothing else — no ccloud, no certificates, no secrets.

set -euo pipefail

CONTAINER_NAME="${RECALL_DEV_DB_CONTAINER:-recall-crdb}"
# Pinned to the cloud cluster's major version so "passes locally" means
# something. Bump this together with the cloud cluster, not independently.
IMAGE="${RECALL_DEV_DB_IMAGE:-cockroachdb/cockroach:latest-v26.2}"
SQL_PORT="${RECALL_DEV_DB_PORT:-26257}"
UI_PORT="${RECALL_DEV_DB_UI_PORT:-8080}"
DSN="postgresql://root@localhost:${SQL_PORT}/defaultdb?sslmode=disable"

# Insecure single-node is deliberate: this cluster holds only throwaway test
# databases, is bound to localhost, and skipping TLS keeps setup to one command.
# Never point RECALL_TEST_DSN at anything real while using --insecure.

die() { echo "error: $*" >&2; exit 1; }

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker Desktop, or see DEV_SETUP.md for the cloud-only path."
    docker info >/dev/null 2>&1 || die "the Docker daemon is not running. Start Docker Desktop and retry."
}

is_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" = "true" ]
}

wait_for_sql() {
    # The container reports "Up" before the SQL layer accepts connections, so
    # poll rather than sleeping a fixed amount.
    local attempts=60
    for _ in $(seq "$attempts"); do
        if docker exec "$CONTAINER_NAME" ./cockroach sql --insecure -e "SELECT 1" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    die "cluster did not accept SQL within ${attempts}s. Check: docker logs $CONTAINER_NAME"
}

cmd_up() {
    require_docker
    if is_running; then
        echo "already running: $CONTAINER_NAME"
    else
        # Remove any stopped container of the same name before recreating.
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        echo "starting $CONTAINER_NAME ($IMAGE)..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "${SQL_PORT}:26257" \
            -p "${UI_PORT}:8080" \
            "$IMAGE" start-single-node --insecure >/dev/null
        wait_for_sql
    fi

    # Migration 002's vector index needs this cluster-wide setting. It is not in
    # the migration itself because a database-scoped cloud role cannot set it —
    # so the local harness sets it here, and the cloud cluster has it set once,
    # out of band.
    docker exec "$CONTAINER_NAME" ./cockroach sql --insecure \
        -e "SET CLUSTER SETTING feature.vector_index.enabled = true" >/dev/null
    echo "ready."
    echo "  SQL:     $DSN"
    echo "  Console: http://localhost:${UI_PORT}"
    echo
    echo "Tests use this by default — just run: pytest"
}

cmd_down() {
    require_docker
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 \
        && echo "removed $CONTAINER_NAME (test data discarded)" \
        || echo "not running: $CONTAINER_NAME"
}

cmd_status() {
    require_docker
    if is_running; then
        local version
        version=$(docker exec "$CONTAINER_NAME" ./cockroach sql --insecure \
            --format=csv -e "SELECT version()" 2>/dev/null \
            | tail -1 | tr -d '"' | cut -d' ' -f1-3)
        echo "running: $CONTAINER_NAME"
        echo "version: $version"
        echo "dsn:     $DSN"
    else
        echo "not running: $CONTAINER_NAME"
        echo "start it with: ./scripts/dev_db.sh up"
        return 1
    fi
}

case "${1:-}" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    reset)  cmd_down; cmd_up ;;
    status) cmd_status ;;
    dsn)    echo "$DSN" ;;
    sql)    require_docker; docker exec -it "$CONTAINER_NAME" ./cockroach sql --insecure ;;
    *)
        # Print the header comment (from line 3 to the first non-comment line)
        # as the usage message, so the docs and the help text cannot drift.
        awk 'NR<3 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
        exit 1
        ;;
esac

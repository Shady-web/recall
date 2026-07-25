#!/usr/bin/env bash
#
# provision.sh — provision the CockroachDB Cloud cluster for Recall.
#
# This script is idempotent: if a cluster named "$CLUSTER_NAME" already exists it
# reports that and exits 0 without changing anything. It provisions a cluster,
# creates a least-privilege service account + API key for automated access, and
# configures a nightly managed backup.
#
# Requirements:
#   - ccloud CLI, authenticated:  ccloud auth login --org "<your-org>"
#   - jq
#
# Verified against the ccloud CLI docs (July 2026):
#   - Global JSON output flag:            -o json
#   - Cluster create (serverless):        ccloud cluster create serverless <name> <region> --cloud <AWS|GCP|AZURE> -o json
#   - Cluster list:                       ccloud cluster list -o json
#   - Connection string:                  ccloud cluster connection-string <name> --sql-user <user> -o json
#   Sources:
#     https://www.cockroachlabs.com/docs/cockroachcloud/ccloud-get-started
#     https://www.cockroachlabs.com/blog/cockroachdb-ai-agents-cli-database-automation/
#
# Flags for service-account creation, api-key creation, and backup configuration
# could NOT be verified from the docs at authoring time. Those commands are
# marked with TODO below — confirm the exact subcommands/flags with
# `ccloud <group> --help` before relying on this script in CI. We do not invent
# flags here.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment)
# ---------------------------------------------------------------------------
CLUSTER_NAME="${CLUSTER_NAME:-recall}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-AWS}"
REGION="${REGION:-us-east-1}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-recall-app}"
SQL_USER="${SQL_USER:-recall_app}"

log()  { printf '\033[1;34m[provision]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[provision]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[provision]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v ccloud >/dev/null 2>&1 || die "ccloud CLI not found on PATH."
command -v jq     >/dev/null 2>&1 || die "jq not found on PATH."

# Confirm we are authenticated. 'ccloud cluster list' fails if we are not.
if ! ccloud cluster list -o json >/dev/null 2>&1; then
    die "ccloud is not authenticated. Run: ccloud auth login --org <your-org>"
fi

# ---------------------------------------------------------------------------
# 1. Idempotency check — does the cluster already exist?
# ---------------------------------------------------------------------------
log "Checking whether cluster '${CLUSTER_NAME}' already exists..."
existing_id="$(
    ccloud cluster list -o json \
        | jq -r --arg name "$CLUSTER_NAME" '.[]? | select(.name == $name) | .id' \
        | head -n1
)"

if [[ -n "${existing_id}" ]]; then
    log "Cluster '${CLUSTER_NAME}' already exists (id: ${existing_id}). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Create the cluster
# ---------------------------------------------------------------------------
log "Creating serverless cluster '${CLUSTER_NAME}' in ${CLOUD_PROVIDER}/${REGION}..."
create_out="$(
    ccloud cluster create serverless "$CLUSTER_NAME" "$REGION" \
        --cloud "$CLOUD_PROVIDER" \
        -o json
)"
cluster_id="$(echo "$create_out" | jq -r '.id // .cluster.id // empty')"
[[ -n "${cluster_id}" ]] || die "Could not determine new cluster id from create output."
log "Created cluster id: ${cluster_id}"

# ---------------------------------------------------------------------------
# 3. Service account + API key (least privilege)
# ---------------------------------------------------------------------------
# We want a service account scoped to operate ONLY this cluster (e.g. the
# "Cluster Operator" role for '${CLUSTER_NAME}'), not org-wide admin.
#
# TODO(verify): confirm the exact subcommands and flags. Expected shape based on
# the CLI's documented "full API surface", but NOT verified against the docs:
#
#   sa_out="$(ccloud service-account create \
#       --name "$SERVICE_ACCOUNT_NAME" \
#       --description "Recall automated app access" \
#       -o json)"
#   sa_id="$(echo "$sa_out" | jq -r '.id')"
#
#   # Grant least-privilege role on THIS cluster only:
#   # TODO(verify): role-grant subcommand/flags.
#   # ccloud service-account grant --service-account "$sa_id" \
#   #     --role CLUSTER_OPERATOR --cluster "$cluster_id"
#
#   key_out="$(ccloud api-key create \
#       --service-account "$sa_id" \
#       -o json)"
#   api_secret="$(echo "$key_out" | jq -r '.secret')"
#
# IMPORTANT: never write the secret to a file or echo it into the repo. Instead,
# print instructions for the operator to store it in a secrets manager. The block
# below shows how we WILL surface it once the flags above are verified:
#
#   warn "API key created. Store it now — it is shown only once."
#   warn "Recommended: put it in your secrets manager, e.g.:"
#   warn "  aws secretsmanager create-secret --name recall/ccloud-api-key --secret-string '<paste>'"
#   # (We intentionally do NOT print \$api_secret to stdout in the final version;
#   #  surface it only through the operator's chosen secret store.)
warn "Service account / API key creation is stubbed pending flag verification."
warn "See the TODO block in $(basename "$0"); create the service account manually"
warn "with least-privilege role on cluster '${CLUSTER_NAME}' and store the key in"
warn "your secrets manager. Do NOT commit it."

# ---------------------------------------------------------------------------
# 4. Nightly managed backup
# ---------------------------------------------------------------------------
# TODO(verify): confirm the backup-configuration subcommand/flags. Expected to be
# under a 'backup' command group on the cluster, but NOT verified:
#
#   # ccloud cluster backup-config update "$cluster_id" \
#   #     --frequency-minutes 1440 \    # nightly
#   #     --retention-days 30 \
#   #     -o json
warn "Nightly backup configuration is stubbed pending flag verification."
warn "Confirm with: ccloud cluster --help  (look for a backup/backup-config group)."

# ---------------------------------------------------------------------------
# 4b. Enable the vector index feature (required by migration 002)
# ---------------------------------------------------------------------------
# Recall's vector index requires this cluster setting. It is applied over SQL,
# not the ccloud control plane:
#     SET CLUSTER SETTING feature.vector_index.enabled = true;
# TODO(verify): the exact way to run one-off SQL via ccloud (e.g. a
# `ccloud cluster sql` subcommand). Until verified, run it with cockroach sql
# against the connection string, e.g.:
#   cockroach sql --url "$conn_str" \
#       -e "SET CLUSTER SETTING feature.vector_index.enabled = true;"
warn "Remember to enable the vector index feature before migrating:"
warn "  SET CLUSTER SETTING feature.vector_index.enabled = true;"

# ---------------------------------------------------------------------------
# 5. Report connection string (no secrets echoed to files)
# ---------------------------------------------------------------------------
log "Fetching connection string template for SQL user '${SQL_USER}'..."
if conn_json="$(ccloud cluster connection-string "$CLUSTER_NAME" --sql-user "$SQL_USER" -o json 2>/dev/null)"; then
    conn_str="$(echo "$conn_json" | jq -r '.connection_string // .connectionString // empty')"
    if [[ -n "${conn_str}" ]]; then
        log "Connection string (password must be set/retrieved separately):"
        log "  ${conn_str}"
        log "Put the full string (with password) into your local .env as CRDB_CONNECTION_STRING."
    fi
else
    warn "Could not fetch connection string automatically; retrieve it from the console/CLI."
fi

log "Done. Cluster '${CLUSTER_NAME}' (${cluster_id}) provisioned."

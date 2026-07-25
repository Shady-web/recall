#!/usr/bin/env bash
#
# audit_export.sh — export CockroachDB Cloud audit logs as JSON.
#
# Pulls cluster audit-log events via the ccloud CLI and writes them to a local
# JSON file. The S3 upload of that file is stubbed with a TODO and wired up in
# Phase 7.
#
# Requirements:
#   - ccloud CLI, authenticated:  ccloud auth login --org "<your-org>"
#   - jq
#
# Verified against the ccloud CLI docs (July 2026):
#   - Audit list:        ccloud audit list --limit <N> --starting-from <RFC3339>
#   - Global JSON flag:  -o json
#   Source: https://www.cockroachlabs.com/blog/cockroachdb-ai-agents-cli-database-automation/

set -euo pipefail

STARTING_FROM="${STARTING_FROM:-$(date -u -d '1 day ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
    || date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)}"
LIMIT="${LIMIT:-1000}"
OUT_DIR="${OUT_DIR:-./audit-exports}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/audit-$(date -u +%Y%m%dT%H%M%SZ).json}"

log()  { printf '\033[1;34m[audit-export]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[audit-export]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[audit-export]\033[0m %s\n' "$*" >&2; exit 1; }

command -v ccloud >/dev/null 2>&1 || die "ccloud CLI not found on PATH."
command -v jq     >/dev/null 2>&1 || die "jq not found on PATH."

mkdir -p "$OUT_DIR"

log "Exporting audit events since ${STARTING_FROM} (limit ${LIMIT})..."
ccloud audit list \
    --limit "$LIMIT" \
    --starting-from "$STARTING_FROM" \
    -o json \
    | jq '.' > "$OUT_FILE"

count="$(jq 'if type == "array" then length else (.events? // [] | length) end' "$OUT_FILE")"
log "Wrote ${count} event(s) to ${OUT_FILE}"

# ---------------------------------------------------------------------------
# S3 upload — wired up in Phase 7.
# ---------------------------------------------------------------------------
# TODO(Phase 7): upload "$OUT_FILE" to the audit-archive S3 bucket, e.g.:
#   aws s3 cp "$OUT_FILE" "s3://${RECALL_AUDIT_BUCKET}/audit/$(basename "$OUT_FILE")" \
#       --region "${AWS_REGION:-us-east-1}"
warn "S3 upload is stubbed (TODO Phase 7). File left locally at ${OUT_FILE}."

#!/usr/bin/env python3
"""Seed ``main`` with believable past incidents, runbooks, and configuration.

This is the corpus the demo agent recalls from. It is written for a plausible
payments/checkout platform so the triage decisions read like real on-call work
rather than toy data.

The set is built around one deliberate trap, because that trap is the whole
point of the demo:

    RB-014 tells you to restart pgbouncer when latency spikes.
    INC-2350 records that doing exactly that dropped 1,400 in-flight payment
    authorizations.

The agent recalls RB-014 and acts on it. Later RB-014 is retracted and replaced
(``--retract-runbook``), and rewind then shows the agent would decide
differently today. Nothing about that outcome is hard-coded: it falls out of
which memories are active at each instant.

Everything is written through the kernel, so every row seeded here carries an
audit entry, exactly like any other write.

Usage:
    python scripts/seed_incidents.py --offline           # local cluster, no cost
    python scripts/seed_incidents.py                     # real Bedrock embeddings
    python scripts/seed_incidents.py --offline --reset   # wipe demo rows first
    python scripts/seed_incidents.py --offline --retract-runbook \\
        --decision <uuid>                                # the "we were wrong" beat
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernel.errors import RecallError  # noqa: E402
from kernel.memory import MemoryKernel  # noqa: E402

logger = logging.getLogger("recall.seed")

#: Marks every row this script writes, so --reset can find them and so the UI
#: can tell seeded history from what the agent learned during the demo.
SEED_TAG = "seed:incident-corpus-v1"

#: The runbook that turns out to be wrong. Referenced by --retract-runbook.
OUTDATED_RUNBOOK_REF = "RB-014"

RETRACTION_REASON = (
    "Superseded after INC-2350: restarting pgbouncer during a live incident drops "
    "in-flight transactions. Replaced by RB-031 (SIGHUP config reload)."
)

REPLACEMENT_RUNBOOK = {
    "ref": "RB-031",
    "kind": "runbook",
    "source": "runbooks/payments/RB-031.md",
    "confidence": 0.97,
    "content": (
        "RB-031 (replaces RB-014): when payments-svc latency exceeds 2s and pgbouncer "
        "reports cl_waiting > 0, NEVER restart the pgbouncer pods during a live "
        "incident — a restart drops every in-flight transaction. Raise max_client_conn "
        "in pgbouncer.ini and apply it with a SIGHUP config reload, which is "
        "zero-downtime and takes effect in under 5 seconds."
    ),
}

#: The seeded corpus. Ordered oldest-first so the timeline scrubber has a
#: sensible left-to-right story to tell.
CORPUS: list[dict[str, Any]] = [
    {
        "ref": "CFG-001",
        "kind": "config",
        "source": "terraform/prod/payments.tf",
        "confidence": 1.0,
        "content": (
            "payments-svc runs 40 replicas in prod us-east-1. Each replica opens a "
            "database connection pool of 10, so the service can demand up to 400 "
            "pooled client connections at full scale."
        ),
    },
    {
        "ref": "CFG-002",
        "kind": "config",
        "source": "ansible/pgbouncer/pgbouncer.ini",
        "confidence": 1.0,
        "content": (
            "pgbouncer max_client_conn is currently set to 200 in prod, with "
            "default_pool_size 25 and pool_mode transaction."
        ),
    },
    {
        "ref": "INC-2211",
        "kind": "incident",
        "source": "postmortems/INC-2211.md",
        "confidence": 0.95,
        "content": (
            "INC-2211: checkout API p99 latency spiked to 4.2s for 38 minutes. Root "
            "cause was connection pool exhaustion in payments-svc — pgbouncer "
            "max_client_conn was 200 while the deploy had scaled to 40 pods x 10 "
            "connections. Clients queued on cl_waiting rather than erroring, which is "
            "why the symptom looked like latency and not failure."
        ),
    },
    {
        "ref": "RB-014",
        "kind": "runbook",
        "source": "runbooks/payments/RB-014.md",
        "confidence": 0.9,
        "content": (
            "RB-014: when payments-svc latency exceeds 2s, immediately restart the "
            "pgbouncer pods. This clears stuck client connections and restores latency "
            "within about 60 seconds."
        ),
    },
    {
        "ref": "RB-021",
        "kind": "runbook",
        "source": "runbooks/payments/RB-021.md",
        "confidence": 0.94,
        "content": (
            "RB-021: scale payments-svc horizontally only after confirming pgbouncer "
            "max_client_conn is at least replicas x pool_size. Scaling past that "
            "ceiling makes connection pool exhaustion worse, not better."
        ),
    },
    {
        "ref": "INC-2247",
        "kind": "incident",
        "source": "postmortems/INC-2247.md",
        "confidence": 0.93,
        "content": (
            "INC-2247: Redis session-store evictions caused cascading 502s at the edge "
            "for 12 minutes. maxmemory was reached and the noeviction policy made "
            "SET fail outright. Fixed by raising maxmemory and switching the policy to "
            "allkeys-lru."
        ),
    },
    {
        "ref": "RB-007",
        "kind": "runbook",
        "source": "runbooks/checkout/RB-007.md",
        "confidence": 0.92,
        "content": (
            "RB-007: checkout feature flags are toggled through LaunchDarkly. The "
            "checkout.v2 kill switch reverts all traffic to the v1 code path in under "
            "30 seconds and is safe to flip during an incident."
        ),
    },
    {
        "ref": "INC-2302",
        "kind": "incident",
        "source": "postmortems/INC-2302.md",
        "confidence": 0.91,
        "content": (
            "INC-2302: payments-svc pods were OOMKilled repeatedly during Black Friday "
            "peak. The 512Mi memory limit was too low for the new JSON serializer, "
            "which buffers whole response bodies. Raised the limit to 1Gi."
        ),
    },
    {
        "ref": "INC-2350",
        "kind": "incident",
        "source": "postmortems/INC-2350.md",
        "confidence": 0.98,
        "content": (
            "INC-2350: during INC-2338 an operator restarted pgbouncer to clear stuck "
            "connections, following RB-014. The restart dropped 1,400 in-flight "
            "payment authorizations, which required two days of manual reconciliation "
            "with the processor. The latency symptom did resolve; the cost of the "
            "remedy exceeded the cost of the incident."
        ),
    },
]


def _metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {"seed": SEED_TAG, "ref": entry["ref"]}


def seed(kernel: MemoryKernel, branch: str = "main") -> list[Any]:
    """Write the corpus onto ``branch``, skipping refs already present.

    Idempotent by ``metadata.ref``: re-running adds only what is missing, so a
    half-finished seed can be resumed and a repeat demo setup is harmless.

    Only **active** rows count as present. That matters because ``reset``
    withdraws the corpus rather than deleting it (nothing in Recall is ever
    hard-deleted), so after a reset every seeded ref still exists — just
    retracted. Treating those as present would make ``--reset`` a one-way trip:
    the corpus would stay withdrawn and re-seeding would silently write
    nothing.
    """
    existing = {
        m.metadata.get("ref")
        for m in kernel.list_memories(branch, limit=500)
        if m.metadata.get("seed") == SEED_TAG and m.status == "active"
    }
    written = []
    for entry in CORPUS:
        if entry["ref"] in existing:
            logger.info("skip %s (already present)", entry["ref"])
            continue
        memory = kernel.remember(
            branch,
            entry["content"],
            kind=entry["kind"],
            source=entry["source"],
            confidence=entry["confidence"],
            metadata=_metadata(entry),
        )
        written.append(memory)
        logger.info("wrote %s -> %s", entry["ref"], memory.id)
    return written


def find_by_ref(kernel: MemoryKernel, branch: str, ref: str):
    """Locate a seeded memory by its human reference (e.g. ``RB-014``)."""
    for memory in kernel.list_memories(branch, limit=500):
        if memory.metadata.get("ref") == ref:
            return memory
    return None


def retract_outdated_runbook(
    kernel: MemoryKernel, branch: str = "main", propagate: bool = True
):
    """The 'we have since learned this was wrong' beat.

    Retracts RB-014 and writes RB-031 in its place. Both are ordinary kernel
    writes: the retraction is recorded with a reason and the old row is kept,
    which is what lets ``explain_decision`` flag any decision that leaned on it.

    ``propagate`` also applies the withdrawal to every **open** branch forked
    from ``branch``, and this is load-bearing rather than convenience. A fork
    freezes its parent at the fork point, so a retraction recorded on ``main``
    afterwards is deliberately invisible to branches that forked earlier — that
    isolation is the whole point of branching, and without propagation a
    decision recorded on a fork would keep reporting its evidence as ``active``
    forever. Withdrawing a bad runbook from the branches still working on it is
    what an operator would actually do, and the kernel expresses it as a
    branch-local override that never touches the parent's row.

    Returns ``(retracted_on_base, replacement_on_base, [affected branch names])``.
    """
    target = find_by_ref(kernel, branch, OUTDATED_RUNBOOK_REF)
    if target is None:
        raise RecallError(
            f"{OUTDATED_RUNBOOK_REF} is not on branch {branch!r}; seed it first"
        )

    replacement = find_by_ref(kernel, branch, REPLACEMENT_RUNBOOK["ref"])
    if replacement is None:
        replacement = kernel.remember(
            branch,
            REPLACEMENT_RUNBOOK["content"],
            kind=REPLACEMENT_RUNBOOK["kind"],
            source=REPLACEMENT_RUNBOOK["source"],
            confidence=REPLACEMENT_RUNBOOK["confidence"],
            metadata=_metadata(REPLACEMENT_RUNBOOK),
        )
        logger.info("wrote %s -> %s", REPLACEMENT_RUNBOOK["ref"], replacement.id)

    if target.status == "active":
        target = kernel.retract(target.id, RETRACTION_REASON, branch=branch)
        logger.info("retracted %s (%s) on %s", OUTDATED_RUNBOOK_REF, target.id, branch)
    else:
        logger.info("%s is already retracted on %s", OUTDATED_RUNBOOK_REF, branch)

    affected: list[str] = []
    if propagate:
        base = kernel.list_branches()
        base_ids = {b.id for b in base if b.name == branch}
        for child in base:
            if child.parent_branch_id not in base_ids or child.status != "open":
                continue
            # Branch-local override: the parent's row is untouched.
            kernel.retract(target.id, RETRACTION_REASON, branch=child.id)
            kernel.remember(
                child.id,
                REPLACEMENT_RUNBOOK["content"],
                kind=REPLACEMENT_RUNBOOK["kind"],
                source=REPLACEMENT_RUNBOOK["source"],
                confidence=REPLACEMENT_RUNBOOK["confidence"],
                metadata=_metadata(REPLACEMENT_RUNBOOK),
            )
            affected.append(child.name)
            logger.info("propagated withdrawal + replacement to %s", child.name)

    return target, replacement, affected


def reset(kernel: MemoryKernel, branch: str = "main") -> int:
    """Retract every seeded row so a rerun starts from a clean narrative.

    Nothing is deleted — this is Recall, so 'reset' means withdrawing the
    memories, not erasing them. Branches forked during earlier demo runs are
    left alone; they are part of the audit record.
    """
    count = 0
    for memory in kernel.list_memories(branch, limit=500):
        if memory.metadata.get("seed") == SEED_TAG and memory.status == "active":
            kernel.retract(memory.id, "demo reset", branch=branch)
            count += 1
    logger.info("retracted %d seeded memories on %s", count, branch)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the fake embedding provider (no AWS calls, no cost)",
    )
    parser.add_argument("--branch", default="main", help="branch to seed")
    parser.add_argument("--dsn", default=None, help="override CRDB_CONNECTION_STRING")
    parser.add_argument(
        "--reset", action="store_true", help="retract existing seeded rows first"
    )
    parser.add_argument(
        "--retract-runbook",
        action="store_true",
        help=f"retract {OUTDATED_RUNBOOK_REF} and write its replacement",
    )
    parser.add_argument(
        "--no-propagate",
        action="store_true",
        help=(
            "retract only on --branch, leaving open forks untouched (they will "
            "keep seeing the withdrawn runbook as active)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from agent.__main__ import build_kernel

    try:
        kernel = build_kernel(args.offline, args.dsn)
        if args.reset:
            reset(kernel, args.branch)
        if args.retract_runbook:
            _, _, affected = retract_outdated_runbook(
                kernel, args.branch, propagate=not args.no_propagate
            )
            print(
                f"withdrew {OUTDATED_RUNBOOK_REF} on {args.branch}"
                + (f" and {len(affected)} open branch(es)" if affected else "")
            )
            return 0
        written = seed(kernel, args.branch)
    except RecallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"seeded {len(written)} new memories on {args.branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

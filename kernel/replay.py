"""Replay and decision provenance for Recall.

=============================================================================
TWO KINDS OF REPLAY, AND WHY BOTH EXIST
=============================================================================

Recall answers two different "what did we know?" questions. They are not the
same question, they can disagree, and **that disagreement is a feature**.

**Logical replay** — :func:`replay_branch_at` — answers
*"what did this branch logically contain at time T?"*
It is pure SQL over the validity columns (``created_at``, ``superseded_at``,
``retracted_at``) plus ``memory_overrides``, bounded by the branch's ancestry.
Because it reads only the *current* rows and reasons about their timestamps, it
works at **any age** — a branch from last month replays exactly as well as one
from five minutes ago. This is the durable, always-available replay, and it is
what provenance and re-runs are built on.

**Physical replay** — :func:`replay_cluster_at` — answers
*"what did the cluster physically look like at time T?"*
It uses ``SET TRANSACTION AS OF SYSTEM TIME``, so it sees the true historical
bytes: rows as they were before any in-place edit, including changes that left
no logical trace. This is the forensic path. It is bounded by MVCC garbage
collection: how far back it reaches is set by the cluster's ``gc.ttlseconds``,
which :func:`replay_window_bounds` reads **live from the zone config on every
call** rather than assuming a value. Clusters differ — a CockroachDB Cloud Basic
cluster reports 4500s (75 min), the self-hosted default is 14400s (4h) — so
never hard-code an expected window. If that live read fails, the code falls back
to a deliberately conservative :data:`_DEFAULT_GC_TTL_SECONDS`, which
under-promises the window: better to refuse a replay the cluster would have
served than to promise one it will reject.

**Where they differ.** Suppose a memory's ``content`` was corrected in place by
an operator. Logical replay shows the corrected text (that is the row Recall
knows about today); physical replay at a pre-correction timestamp shows the
original bytes. Logical replay is the system of record's own account of itself;
physical replay is the ground truth of what was stored. When they disagree, the
difference is exactly the evidence an audit needs.

Note on the audit invariant: an ``AS OF SYSTEM TIME`` transaction is read-only,
so physical replay cannot write its audit row inside the same transaction. It
writes the audit row immediately afterwards in its own transaction. This is the
one place the kernel cannot honour same-transaction auditing, and it is a
CockroachDB constraint rather than a choice.

This module stays free of agent and Bedrock specifics: :func:`rewind_and_rerun`
takes an injected callable. The kernel remains pure.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, NamedTuple

import psycopg
from psycopg.rows import dict_row

from kernel import audit
from kernel.branching import effective_status_sql, resolve_ancestry
from kernel.db import Database
from kernel.errors import NotFoundError, ReplayWindowExpiredError
from kernel.memory import MEMORY_COLUMNS, resolve_branch
from kernel.models import (
    AgentDecision,
    ContributingMemory,
    Decision,
    DecisionExplanation,
    Memory,
    MemoryAvailabilityDiff,
    MemoryRef,
    ReplayWindow,
    RerunContext,
    RerunResult,
    RewindSummary,
)

logger = logging.getLogger("recall.kernel.replay")

# Fallback used ONLY when the live zone config cannot be read (e.g. a restricted
# role). Deliberately far shorter than any real cluster's gc.ttlseconds so a
# failed read under-promises the window instead of over-promising it: claiming a
# window wider than reality would wave through timestamps the cluster then
# refuses, which is the failure this guard exists to prevent. Observed real
# values vary widely (a CockroachDB Cloud Basic cluster reports 4500s; the
# self-hosted default is 14400s), so there is no safe "typical" value to assume.
_DEFAULT_GC_TTL_SECONDS = 600

_GC_TTL_RE = re.compile(r"gc\.ttlseconds\s*=\s*(\d+)")


# --------------------------------------------------------------------------
# Replay window (physical replay only)
# --------------------------------------------------------------------------


def replay_window_bounds(db: Database) -> ReplayWindow:
    """Report the range over which physical replay is currently safe.

    Logical replay is *not* limited by this — it works at any age.
    """

    def work(conn: psycopg.Connection) -> ReplayWindow:
        ttl = _DEFAULT_GC_TTL_SECONDS
        try:
            row = conn.execute(
                "SHOW ZONE CONFIGURATION FOR TABLE memories"
            ).fetchone()
            match = _GC_TTL_RE.search(str(row[1])) if row is not None else None
            if match:
                ttl = int(match.group(1))
            else:
                # Read succeeded but yielded no gc.ttlseconds — same silent-narrowing
                # risk as an outright failure, so it warns too.
                logger.warning(
                    "could not parse gc.ttlseconds from the zone configuration for "
                    "table 'memories'; using the conservative fallback of %ds. "
                    "Physical replay will refuse anything older than that, which may "
                    "be far narrower than this cluster's real GC window.",
                    _DEFAULT_GC_TTL_SECONDS,
                )
        except Exception as exc:
            logger.warning(
                "live read of gc.ttlseconds failed (%s: %s); using the conservative "
                "fallback of %ds. Physical replay will refuse anything older than "
                "that, so replay_cluster_at() may raise ReplayWindowExpiredError for "
                "timestamps it previously served. Check that this role can run "
                "SHOW ZONE CONFIGURATION.",
                type(exc).__name__,
                exc,
                _DEFAULT_GC_TTL_SECONDS,
            )
        now = conn.execute("SELECT now()").fetchone()[0]
        return ReplayWindow(
            gc_ttl_seconds=ttl,
            earliest=now - timedelta(seconds=ttl),
            latest=now,
        )

    return db.run_in_transaction(work, read_only=True)


# --------------------------------------------------------------------------
# Logical replay — durable, any age
# --------------------------------------------------------------------------


def _clamped_segments(db: Database, conn: psycopg.Connection, branch_id, t: datetime):
    """Ancestry segments with every visibility bound clamped to ``t``.

    A segment's bound is ``min(fork bound, t)``; the branch's own segment (bound
    ``None`` = live) becomes simply ``t``. Clamping is what turns "what does this
    branch see now" into "what did it see at T".
    """
    segments = resolve_ancestry(db, branch_id, conn=conn)
    clamped = []
    for seg in segments:
        bound = t if seg.visible_as_of is None else min(seg.visible_as_of, t)
        clamped.append((seg.branch_id, bound, seg.depth))
    return segments, clamped


def replay_branch_at(
    db: Database,
    actor: str,
    branch: str | uuid.UUID,
    t: datetime,
    *,
    kind: str | None = None,
    status: str | None = "active",
    audit_op: str = "replay_branch",
) -> list[Memory]:
    """Reconstruct what ``branch`` logically contained at wall time ``t``.

    Pure SQL over the validity columns, so this works for a branch of **any
    age** — it is not bounded by MVCC garbage collection.
    """

    def work(conn: psycopg.Connection) -> list[Memory]:
        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            _, clamped = _clamped_segments(db, conn, b["id"], t)

            anc_values = ", ".join(
                ["(%s::UUID, %s::TIMESTAMPTZ, %s::INT)"] * len(clamped)
            )
            status_expr = effective_status_sql("m")
            where = ["m.created_at <= anc.visible_as_of"]
            params: list[Any] = []
            for bid, bound, depth in clamped:
                params.extend([bid, bound, depth])
            if status is not None:
                where.append(f"{status_expr} = %s")
            if kind is not None:
                where.append("m.kind = %s")

            sql = (
                f"WITH anc (branch_id, visible_as_of, depth) AS (VALUES {anc_values}) "
                f"SELECT m.id, m.branch_id, m.kind, m.content, m.source, m.confidence, "
                f"       {status_expr} AS status, m.superseded_by, m.metadata, m.created_at "
                f"  FROM memories m "
                f"  JOIN anc ON anc.branch_id = m.branch_id "
                f" WHERE {' AND '.join(where)} "
                f" ORDER BY m.created_at"
            )
            if status is not None:
                params.append(status)
            if kind is not None:
                params.append(kind)

            cur.execute(sql, params)
            rows = cur.fetchall()

            audit.record(
                conn,
                actor=actor,
                op=audit_op,
                target_type="branch",
                target_id=b["id"],
                payload={
                    "replay_at": t.isoformat(),
                    "mode": "logical",
                    "count": len(rows),
                    "kind": kind,
                    "status": status,
                },
            )
            return [Memory.model_validate(r) for r in rows]

    return db.run_in_transaction(work)


# --------------------------------------------------------------------------
# Physical replay — forensic, GC-bounded
# --------------------------------------------------------------------------


def replay_cluster_at(
    db: Database,
    actor: str,
    t: datetime,
    *,
    branch: str | uuid.UUID | None = None,
) -> list[Memory]:
    """Read true historical cluster state at ``t`` via ``AS OF SYSTEM TIME``.

    Sees the physical bytes as they were — including rows later changed in place
    in ways that left no logical trace.

    Raises :class:`ReplayWindowExpiredError` if ``t`` is older than the MVCC GC
    window, rather than returning wrong or empty data. (The raw engine error in
    that case is actively misleading — e.g. ``database ... does not exist`` —
    which is precisely why this guard exists.)
    """
    window = replay_window_bounds(db)
    if t < window.earliest:
        raise ReplayWindowExpiredError(
            f"cannot physically replay at {t.isoformat()}: older than the MVCC "
            f"garbage-collection window (gc.ttlseconds={window.gc_ttl_seconds}). "
            f"Safe physical-replay range is {window.earliest.isoformat()} .. "
            f"{window.latest.isoformat()}. Use replay_branch_at() for logical "
            f"replay, which works at any age."
        )
    if t > window.latest:
        raise ReplayWindowExpiredError(
            f"cannot replay at {t.isoformat()}: that is in the future "
            f"(cluster time is {window.latest.isoformat()})."
        )

    branch_id = None
    if branch is not None:
        with db.transaction(read_only=True) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                branch_id = resolve_branch(cur, branch)["id"]

    # An AOST transaction is read-only, so the audit row cannot be written
    # inside it (see the module docstring).
    with db.pool.connection() as conn:
        conn.read_only = True
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SET TRANSACTION AS OF SYSTEM TIME '{t.isoformat()}'")
                sql = f"SELECT {MEMORY_COLUMNS} FROM memories"
                params: list[Any] = []
                if branch_id is not None:
                    sql += " WHERE branch_id = %s"
                    params.append(branch_id)
                sql += " ORDER BY created_at"
                cur.execute(sql, params)
                rows = cur.fetchall()
        conn.read_only = False

    def audit_work(conn: psycopg.Connection) -> None:
        audit.record(
            conn,
            actor=actor,
            op="replay_cluster",
            target_type="branch" if branch_id is not None else "cluster",
            target_id=branch_id,
            payload={
                "replay_at": t.isoformat(),
                "mode": "physical",
                "gc_ttl_seconds": window.gc_ttl_seconds,
                "count": len(rows),
            },
        )

    db.run_in_transaction(audit_work)
    return [Memory.model_validate(r) for r in rows]


# --------------------------------------------------------------------------
# Decision provenance
# --------------------------------------------------------------------------


def _load_decision(cur: psycopg.Cursor, decision_id: str | uuid.UUID) -> dict:
    cur.execute(
        "SELECT d.id, d.branch_id, d.agent_id, d.input_hash, d.action, d.rationale, "
        "       d.outcome, d.created_at, b.name AS branch_name "
        "  FROM decisions d JOIN branches b ON b.id = d.branch_id "
        " WHERE d.id = %s",
        (decision_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"decision not found: {decision_id}")
    return row


def explain_decision(
    db: Database, actor: str, decision_id: str | uuid.UUID
) -> DecisionExplanation:
    """Explain a decision: which memories drove it, and what has changed since.

    Each contributing memory carries its similarity and rank **as recorded at
    decision time**, plus its **current** status. Any memory since superseded or
    retracted is flagged — both on the memory and at the top level — because
    "the agent acted on X, which we have since learned was wrong" is the
    question this exists to answer.
    """

    def work(conn: psycopg.Connection) -> DecisionExplanation:
        with conn.cursor(row_factory=dict_row) as cur:
            d = _load_decision(cur, decision_id)
            segments = resolve_ancestry(db, d["branch_id"], conn=conn)

            anc_values = ", ".join(
                ["(%s::UUID, %s::TIMESTAMPTZ, %s::INT)"] * len(segments)
            )
            params: list[Any] = []
            for seg in segments:
                params.extend([seg.branch_id, seg.visible_as_of, seg.depth])
            params.append(d["id"])

            status_expr = effective_status_sql("m")
            cur.execute(
                f"WITH anc (branch_id, visible_as_of, depth) AS (VALUES {anc_values}) "
                f"SELECT dm.memory_id, dm.similarity, dm.rank, "
                f"       m.content, m.kind, m.source, m.confidence, m.branch_id, "
                f"       m.created_at, m.superseded_by, m.superseded_at, m.retracted_at, "
                f"       {status_expr} AS status_now "
                f"  FROM decision_memories dm "
                f"  JOIN memories m ON m.id = dm.memory_id "
                f"  LEFT JOIN anc ON anc.branch_id = m.branch_id "
                f" WHERE dm.decision_id = %s "
                f" ORDER BY dm.rank",
                params,
            )
            contributions: list[ContributingMemory] = []
            for r in cur.fetchall():
                status_now = r["status_now"]
                superseded = status_now == "superseded"
                retracted = status_now == "retracted"
                contributions.append(
                    ContributingMemory(
                        memory_id=r["memory_id"],
                        content=r["content"],
                        kind=r["kind"],
                        source=r["source"],
                        confidence=r["confidence"],
                        branch_id=r["branch_id"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                        rank=r["rank"],
                        status_now=status_now,
                        invalidated=superseded or retracted,
                        superseded=superseded,
                        retracted=retracted,
                        superseded_by=r["superseded_by"],
                        superseded_at=r["superseded_at"],
                        retracted_at=r["retracted_at"],
                    )
                )

            invalidated = [c.memory_id for c in contributions if c.invalidated]
            explanation = DecisionExplanation(
                decision=Decision.model_validate(d),
                branch_id=d["branch_id"],
                branch_name=d["branch_name"],
                memories=contributions,
                has_invalidated_memories=bool(invalidated),
                invalidated_count=len(invalidated),
                invalidated_memory_ids=invalidated,
            )

            audit.record(
                conn,
                actor=actor,
                op="explain_decision",
                target_type="decision",
                target_id=d["id"],
                payload={
                    "contributing_count": len(contributions),
                    "invalidated_count": len(invalidated),
                    "invalidated_memory_ids": [str(i) for i in invalidated],
                },
            )
            return explanation

    return db.run_in_transaction(work)


# --------------------------------------------------------------------------
# Rewind and re-run
# --------------------------------------------------------------------------


def _normalize(result: Any) -> AgentDecision:
    """Accept a str, dict, AgentDecision, or any object with ``.action``."""
    if isinstance(result, AgentDecision):
        return result
    if isinstance(result, str):
        return AgentDecision(action=result)
    if isinstance(result, dict):
        return AgentDecision(
            action=str(result["action"]), rationale=result.get("rationale")
        )
    action = getattr(result, "action", None)
    if action is None:
        raise TypeError(
            "agent callable must return a str, a dict with 'action', an "
            "AgentDecision, or an object with an .action attribute"
        )
    return AgentDecision(action=str(action), rationale=getattr(result, "rationale", None))


def _ref(m: Memory) -> MemoryRef:
    return MemoryRef(
        memory_id=m.id, content=m.content, kind=m.kind, status=m.status
    )


class _RewindState(NamedTuple):
    """What both rewind paths need before they diverge."""

    decision: dict[str, Any]
    contributing: list[uuid.UUID]
    then: list[Memory]
    now: list[Memory]
    window: ReplayWindow
    diff: MemoryAvailabilityDiff


def _rewind_state(db: Database, actor: str, decision_id: str | uuid.UUID) -> _RewindState:
    """Reconstruct the shared rewind context: the decision, its provenance, and
    the branch as it was at decision time vs now.

    Shared by :func:`rewind_summary` (which stops here) and
    :func:`rewind_and_rerun` (which then runs an agent against it), so the two
    can never drift on what "then" and "now" mean.
    """
    with db.transaction(read_only=True) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            d = _load_decision(cur, decision_id)
            cur.execute(
                "SELECT memory_id FROM decision_memories WHERE decision_id = %s "
                "ORDER BY rank",
                (d["id"],),
            )
            contributing = [r["memory_id"] for r in cur.fetchall()]

    then = replay_branch_at(
        db, actor, d["branch_id"], d["created_at"], audit_op="rewind_replay"
    )
    now_window = replay_window_bounds(db)
    now_memories = replay_branch_at(
        db, actor, d["branch_id"], now_window.latest, audit_op="rewind_replay"
    )

    then_ids = {m.id for m in then}
    now_ids = {m.id for m in now_memories}
    diff = MemoryAvailabilityDiff(
        only_then=[_ref(m) for m in then if m.id not in now_ids],
        only_now=[_ref(m) for m in now_memories if m.id not in then_ids],
        common_count=len(then_ids & now_ids),
        then_count=len(then),
        now_count=len(now_memories),
    )
    return _RewindState(d, contributing, then, now_memories, now_window, diff)


def rewind_summary(
    db: Database,
    actor: str,
    decision_id: str | uuid.UUID,
) -> RewindSummary:
    """Rewind to a decision and report what changed — **without re-running any agent**.

    This is :func:`rewind_and_rerun` minus the agent: the branch is rebuilt with
    logical replay (so it works at any age) and the then-vs-now memory
    availability diff is returned, but nothing is invoked and no new decision is
    produced. Callers that must not trigger a model call as a side effect of a
    read should use this.
    """
    state = _rewind_state(db, actor, decision_id)
    d, diff = state.decision, state.diff

    summary = RewindSummary(
        decision_id=d["id"],
        branch_id=d["branch_id"],
        branch_name=d["branch_name"],
        decision_at=d["created_at"],
        replayed_at=d["created_at"],
        action=d["action"],
        rationale=d["rationale"],
        memories_at_decision=[_ref(m) for m in state.then],
        memory_diff=diff,
        contributing_memory_ids=state.contributing,
    )

    def audit_work(conn: psycopg.Connection) -> None:
        audit.record(
            conn,
            actor=actor,
            op="rewind_summary",
            target_type="decision",
            target_id=d["id"],
            payload={
                "replayed_at": summary.replayed_at.isoformat(),
                "memories_then": diff.then_count,
                "memories_now": diff.now_count,
                "only_then": len(diff.only_then),
                "only_now": len(diff.only_now),
            },
        )

    db.run_in_transaction(audit_work)
    return summary


def rewind_and_rerun(
    db: Database,
    actor: str,
    decision_id: str | uuid.UUID,
    agent: Callable[[RerunContext], Any],
    *,
    as_of: datetime | None = None,
) -> RerunResult:
    """Reconstruct the branch as of a decision and re-run an agent against it.

    The branch state is rebuilt with **logical** replay, so this works
    regardless of the decision's age. ``agent`` is injected and receives a
    :class:`RerunContext`; the kernel knows nothing about how it reaches a
    decision.

    ``as_of`` selects which reconstructed state the agent runs against:

    * **default (the decision's timestamp)** — faithful replay. A deterministic
      agent reproduces the original action, which is the fidelity check; a
      *changed* agent reveals how a new version would have handled a past
      incident.
    * **a later time (e.g. now)** — "given what we have learned since, would it
      still decide this?". This is what flips the action when a supporting
      memory has been retracted or superseded.

    Either way the returned ``memory_diff`` always compares decision time
    against now, so the caller can see exactly what changed.
    """
    state = _rewind_state(db, actor, decision_id)
    d, now_window, diff = state.decision, state.window, state.diff

    decision = Decision.model_validate(d)
    evaluated_at = as_of if as_of is not None else d["created_at"]
    if as_of is None:
        context_memories = state.then
    elif as_of == now_window.latest:
        context_memories = state.now
    else:
        context_memories = replay_branch_at(
            db, actor, d["branch_id"], as_of, audit_op="rewind_replay"
        )
    new = _normalize(
        agent(
            RerunContext(
                decision=decision, memories=context_memories, replayed_at=evaluated_at
            )
        )
    )

    result = RerunResult(
        decision_id=d["id"],
        branch_id=d["branch_id"],
        branch_name=d["branch_name"],
        decision_at=d["created_at"],
        evaluated_at=evaluated_at,
        old_action=d["action"],
        old_rationale=d["rationale"],
        new_action=new.action,
        new_rationale=new.rationale,
        action_changed=new.action != d["action"],
        memory_diff=diff,
        contributing_memory_ids=state.contributing,
    )

    def audit_work(conn: psycopg.Connection) -> None:
        audit.record(
            conn,
            actor=actor,
            op="rewind_and_rerun",
            target_type="decision",
            target_id=d["id"],
            payload={
                "old_action": result.old_action,
                "new_action": result.new_action,
                "action_changed": result.action_changed,
                "evaluated_at": evaluated_at.isoformat(),
                "memories_then": diff.then_count,
                "memories_now": diff.now_count,
            },
        )

    db.run_in_transaction(audit_work)
    return result

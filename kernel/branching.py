"""Branching engine for Recall — fork, commit, discard, diff, ancestry.

=============================================================================
SEMANTICS THIS MODULE IMPLEMENTS
=============================================================================

* ``fork(parent_branch, name)`` creates a child branch whose ``fork_point_ts``
  is the current cluster timestamp. The child starts with no memories of its
  own; everything it "knows" at birth is inherited from the parent.

* **Reads on a branch see**: that branch's own memories, plus the parent's state
  as of ``fork_point_ts``, recursively up the ancestry chain. Concretely, for a
  chain ``B -> P1 -> P2``, a read on ``B`` sees:
      - all of ``B``'s memories (no time bound — B's own writes are live), plus
      - ``P1``'s memories created at or before ``B.fork_point_ts``, plus
      - ``P2``'s memories created at or before ``min(P1.fork_point_ts,
        B.fork_point_ts)``.
  Each ancestor's bound is the *minimum* of the fork points below it, so an
  ancestor can never leak content that came into existence after the descendant
  branched away.

* **Writes on a branch never affect the parent.** A branch that supersedes or
  retracts an *inherited* memory does NOT touch the ancestor's row. It records a
  branch-local entry in ``memory_overrides`` that shadows that memory for this
  branch only. The ancestor keeps seeing its own memory as active.

* ``commit(branch)`` replays the branch's memories onto the parent as **new
  rows** (never moving rows between branches), detects conflicts, and marks the
  branch ``committed``. If any conflict is found the commit is a **no-op**:
  nothing is replayed, the branch stays open, and the conflicts come back as
  structured data for the caller to decide on.

* ``discard(branch)`` marks the branch ``discarded``. Nothing is ever
  hard-deleted — discarded branches and their memories remain readable for audit
  and replay.

* ``diff(branch_a, branch_b)`` reports the memories added, superseded, and
  retracted on each side.

* **Conflict rule**: if a memory was superseded (or retracted) on the parent
  *after* the fork point, and the branch also modified that same memory, that is
  a conflict. Concurrent commits are serialized by CockroachDB's SERIALIZABLE
  isolation, so of two racing commits exactly one wins cleanly and the other
  observes the parent's new state and reports a conflict.

=============================================================================
HOW ANCESTRY IS RESOLVED (and why it is not AS OF SYSTEM TIME)
=============================================================================

CONTEXT.md §7 assumed a recursive CTE plus ``AS OF SYSTEM TIME`` at each fork
point. That is not implementable on CockroachDB, verified empirically:

  * AOST must be attached to a **top-level statement**. Inside a CTE, a
    sub-select, or one arm of a UNION it fails with
    ``AS OF SYSTEM TIME must be provided on a top-level statement``. One
    statement therefore cannot read different ancestry segments at different
    timestamps — which is precisely what per-fork-point AOST requires.
  * AOST cannot look back past MVCC garbage collection
    (``gc.ttlseconds`` default 14400 = 4h), so any branch outliving the GC
    window would become unreadable.

Instead we use **logical time-travel**: each ancestry segment is bounded by
``created_at <= visible_as_of``, and status changes carry their own timestamps
(``superseded_at`` / ``retracted_at``, added in migration 003) so that "status as
of T" is computable in SQL. This is exact for our append-only memory rows, is not
GC-bounded, and keeps the branch-scoped read in a single index-accelerated
statement.

One structural note, also verified empirically: expressing the ancestry as an
inline ``branch_id IN (SELECT id FROM ancestry_cte)`` **defeats the vector
index** (the planner falls back to full scans). Passing the resolved branch ids
as a literal list preserves the index (prefix spans on ``branch_id``). So
:func:`resolve_ancestry` runs the recursive CTE over the tiny ``branches`` table
first, and the memory read is then one index-accelerated statement over the
resolved chain. See :mod:`kernel.recall`.
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from kernel import audit
from kernel.db import Database
from kernel.errors import InvalidStateError, NotFoundError, ReadOnlyError
from kernel.memory import MEMORY_COLUMNS, resolve_branch
from kernel.models import (
    AncestrySegment,
    Branch,
    BranchDiff,
    BranchSideDiff,
    CommitResult,
    MemoryConflict,
)

# Recursive CTE walking a branch to the root, computing each ancestor's
# visibility bound. The branch itself gets bound NULL (live, unbounded); each
# ancestor gets LEAST(fork point below it, bound below it) so bounds decrease
# monotonically toward the root.
_ANCESTRY_CTE = """
WITH RECURSIVE ancestry AS (
    SELECT id, name, parent_branch_id, fork_point_ts, 0 AS depth,
           NULL::TIMESTAMPTZ AS visible_as_of
      FROM branches
     WHERE id = %s
    UNION ALL
    SELECT b.id, b.name, b.parent_branch_id, b.fork_point_ts, a.depth + 1,
           LEAST(a.fork_point_ts, COALESCE(a.visible_as_of, a.fork_point_ts))
      FROM branches b
      JOIN ancestry a ON b.id = a.parent_branch_id
)
SELECT id AS branch_id, name, depth, visible_as_of FROM ancestry ORDER BY depth
"""


def effective_status_sql(
    alias: str, anc_alias: str = "anc", anc2_alias: str = "anc2"
) -> str:
    """SQL expression for a memory's status as of its ancestry segment's bound.

    A branch-local override from the *nearest* ancestry branch wins; otherwise
    the status is derived from the row's own ``superseded_at`` / ``retracted_at``
    compared against that segment's ``visible_as_of`` bound (``NULL`` meaning
    live/unbounded).

    Shared by :mod:`kernel.recall` (what a branch sees now) and
    :mod:`kernel.replay` (what it saw at time T) so both agree on semantics —
    replay simply passes bounds clamped to the replay timestamp.
    """
    return f"""
COALESCE(
    (SELECT o.status
       FROM memory_overrides o
       JOIN {anc_alias} {anc2_alias} ON {anc2_alias}.branch_id = o.branch_id
      WHERE o.memory_id = {alias}.id
        AND ({anc2_alias}.visible_as_of IS NULL
             OR o.created_at <= {anc2_alias}.visible_as_of)
      ORDER BY {anc2_alias}.depth
      LIMIT 1),
    CASE
        WHEN {alias}.retracted_at IS NOT NULL
             AND ({anc_alias}.visible_as_of IS NULL
                  OR {alias}.retracted_at <= {anc_alias}.visible_as_of)
            THEN 'retracted'
        WHEN {alias}.superseded_at IS NOT NULL
             AND ({anc_alias}.visible_as_of IS NULL
                  OR {alias}.superseded_at <= {anc_alias}.visible_as_of)
            THEN 'superseded'
        ELSE 'active'
    END
)
"""


def resolve_ancestry(
    db: Database, branch: str | uuid.UUID, conn: psycopg.Connection | None = None
) -> list[AncestrySegment]:
    """Return the ordered ancestry chain for ``branch``.

    Index 0 is the branch itself (``visible_as_of`` is ``None`` — its own writes
    are always visible); each subsequent entry is one step closer to the root,
    carrying the timestamp bound at which that ancestor is visible.
    """

    def work(c: psycopg.Connection) -> list[AncestrySegment]:
        with c.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            cur.execute(_ANCESTRY_CTE, (b["id"],))
            rows = cur.fetchall()
        return [AncestrySegment.model_validate(r) for r in rows]

    if conn is not None:
        return work(conn)
    return db.run_in_transaction(work, read_only=True)


def fork(
    db: Database,
    actor: str,
    parent_branch: str | uuid.UUID,
    name: str,
    *,
    read_only: bool = False,
) -> Branch:
    """Create a child branch of ``parent_branch`` forked at the current time.

    ``fork_point_ts`` is taken from the cluster (``now()``), not the client, so
    concurrent forks agree on ordering.
    """
    if read_only:
        raise ReadOnlyError("kernel is in read-only mode; fork is blocked")

    def work(conn: psycopg.Connection) -> Branch:
        with conn.cursor(row_factory=dict_row) as cur:
            parent = resolve_branch(cur, parent_branch)
            cur.execute("SELECT status FROM branches WHERE id = %s", (parent["id"],))
            if cur.fetchone()["status"] == "discarded":
                raise InvalidStateError(
                    f"cannot fork from discarded branch {parent['name']!r}"
                )
            cur.execute(
                "INSERT INTO branches (name, parent_branch_id, fork_point_ts, "
                "                      status, created_by) "
                "VALUES (%s, %s, now(), 'open', %s) "
                "RETURNING id, name, parent_branch_id, fork_point_ts, status, "
                "          created_by, created_at",
                (name, parent["id"], actor),
            )
            row = cur.fetchone()
            audit.record(
                conn,
                actor=actor,
                op="fork",
                target_type="branch",
                target_id=row["id"],
                payload={
                    "name": name,
                    "parent_branch_id": str(parent["id"]),
                    "parent_name": parent["name"],
                    "fork_point_ts": row["fork_point_ts"].isoformat(),
                },
            )
            return Branch.model_validate(row)

    return db.run_in_transaction(work)


def discard(
    db: Database,
    actor: str,
    branch: str | uuid.UUID,
    reason: str = "",
    *,
    read_only: bool = False,
) -> Branch:
    """Mark ``branch`` discarded. Nothing is hard-deleted."""
    if read_only:
        raise ReadOnlyError("kernel is in read-only mode; discard is blocked")

    def work(conn: psycopg.Connection) -> Branch:
        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            cur.execute(
                "UPDATE branches SET status = 'discarded' WHERE id = %s "
                "RETURNING id, name, parent_branch_id, fork_point_ts, status, "
                "          created_by, created_at",
                (b["id"],),
            )
            row = cur.fetchone()
            audit.record(
                conn,
                actor=actor,
                op="discard",
                target_type="branch",
                target_id=b["id"],
                payload={"name": b["name"], "reason": reason},
            )
            return Branch.model_validate(row)

    return db.run_in_transaction(work)


def _detect_conflicts(
    cur: psycopg.Cursor, branch_id: uuid.UUID, parent_id: uuid.UUID, fork_ts: Any
) -> list[MemoryConflict]:
    """Find memories this branch modified that the parent also changed post-fork.

    A branch "modified" an inherited memory iff it holds an override for it.
    The parent "changed" it iff the parent's own row was superseded or retracted
    strictly after the fork point.
    """
    cur.execute(
        """
        SELECT m.id, m.content, m.status AS parent_status,
               o.status AS branch_status,
               GREATEST(COALESCE(m.superseded_at, m.retracted_at),
                        COALESCE(m.retracted_at,  m.superseded_at)) AS parent_changed_at
          FROM memory_overrides o
          JOIN memories m ON m.id = o.memory_id
         WHERE o.branch_id = %s
           AND m.branch_id = %s
           AND (m.superseded_at > %s OR m.retracted_at > %s)
        """,
        (branch_id, parent_id, fork_ts, fork_ts),
    )
    return [
        MemoryConflict(
            memory_id=r["id"],
            content=r["content"],
            branch_status=r["branch_status"],
            parent_status=r["parent_status"],
            parent_changed_at=r["parent_changed_at"],
            fork_point_ts=fork_ts,
        )
        for r in cur.fetchall()
    ]


def commit(
    db: Database,
    actor: str,
    branch: str | uuid.UUID,
    *,
    read_only: bool = False,
) -> CommitResult:
    """Replay ``branch``'s memories onto its parent and mark it committed.

    Returns a :class:`CommitResult`. If conflicts are detected the commit is a
    **no-op** — nothing is replayed, the branch stays open, and the conflicts are
    returned for the caller to resolve. Conflicts are data, not exceptions.

    The whole operation runs in one SERIALIZABLE transaction, so two racing
    commits against the same parent cannot both apply: one wins, the other sees
    the parent's updated state and reports a conflict.
    """
    if read_only:
        raise ReadOnlyError("kernel is in read-only mode; commit is blocked")

    def work(conn: psycopg.Connection) -> CommitResult:
        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            cur.execute(
                "SELECT id, name, parent_branch_id, fork_point_ts, status "
                "FROM branches WHERE id = %s",
                (b["id"],),
            )
            br = cur.fetchone()
            if br["parent_branch_id"] is None:
                raise InvalidStateError(f"branch {br['name']!r} is a root; nothing to commit into")
            if br["status"] != "open":
                raise InvalidStateError(
                    f"branch {br['name']!r} is {br['status']}; only open branches can be committed"
                )
            parent_id = br["parent_branch_id"]
            fork_ts = br["fork_point_ts"]

            conflicts = _detect_conflicts(cur, br["id"], parent_id, fork_ts)
            if conflicts:
                # No-op commit: leave the branch open and write nothing but the
                # audit trail of the attempt.
                audit.record(
                    conn,
                    actor=actor,
                    op="commit",
                    target_type="branch",
                    target_id=br["id"],
                    payload={
                        "committed": False,
                        "conflict_count": len(conflicts),
                        "conflict_memory_ids": [str(c.memory_id) for c in conflicts],
                    },
                )
                return CommitResult(
                    branch_id=br["id"], committed=False, conflicts=conflicts
                )

            # 1. Replay this branch's own memories onto the parent as NEW rows.
            cur.execute(
                f"SELECT {MEMORY_COLUMNS}, embedding FROM memories "
                f"WHERE branch_id = %s AND status = 'active' ORDER BY created_at",
                (br["id"],),
            )
            replayed: list[uuid.UUID] = []
            for m in cur.fetchall():
                cur.execute(
                    "INSERT INTO memories (branch_id, kind, content, embedding, source, "
                    "                      confidence, metadata, origin_memory_id) "
                    "SELECT %s, kind, content, embedding, source, confidence, metadata, id "
                    "FROM memories WHERE id = %s RETURNING id",
                    (parent_id, m["id"]),
                )
                replayed.append(cur.fetchone()["id"])

            # 2. Apply this branch's overrides to the parent's own rows.
            cur.execute(
                "SELECT memory_id, status, superseded_by, reason FROM memory_overrides "
                "WHERE branch_id = %s",
                (br["id"],),
            )
            applied: list[uuid.UUID] = []
            for o in cur.fetchall():
                if o["status"] == "superseded":
                    cur.execute(
                        "UPDATE memories SET status = 'superseded', superseded_at = now(), "
                        "superseded_by = %s WHERE id = %s AND branch_id = %s",
                        (o["superseded_by"], o["memory_id"], parent_id),
                    )
                else:
                    cur.execute(
                        "UPDATE memories SET status = 'retracted', retracted_at = now() "
                        "WHERE id = %s AND branch_id = %s",
                        (o["memory_id"], parent_id),
                    )
                if cur.rowcount:
                    applied.append(o["memory_id"])

            cur.execute("UPDATE branches SET status = 'committed' WHERE id = %s", (br["id"],))
            audit.record(
                conn,
                actor=actor,
                op="commit",
                target_type="branch",
                target_id=br["id"],
                payload={
                    "committed": True,
                    "parent_branch_id": str(parent_id),
                    "replayed_count": len(replayed),
                    "replayed_memory_ids": [str(i) for i in replayed],
                    "applied_override_ids": [str(i) for i in applied],
                },
            )
            return CommitResult(
                branch_id=br["id"],
                committed=True,
                replayed_memory_ids=replayed,
                applied_override_ids=applied,
            )

    return db.run_in_transaction(work)


def _side_diff(cur: psycopg.Cursor, branch_id: uuid.UUID, name: str) -> BranchSideDiff:
    """Summarize what one branch did: memories added, superseded, retracted."""
    cur.execute(
        "SELECT id FROM memories WHERE branch_id = %s ORDER BY created_at", (branch_id,)
    )
    added = [r["id"] for r in cur.fetchall()]

    # Status changes this branch made: to inherited memories (overrides) and to
    # its own rows (timestamps on the row itself).
    cur.execute(
        "SELECT memory_id, status FROM memory_overrides WHERE branch_id = %s", (branch_id,)
    )
    superseded = []
    retracted = []
    for r in cur.fetchall():
        (superseded if r["status"] == "superseded" else retracted).append(r["memory_id"])

    cur.execute(
        "SELECT id, status FROM memories WHERE branch_id = %s "
        "AND status IN ('superseded', 'retracted')",
        (branch_id,),
    )
    for r in cur.fetchall():
        (superseded if r["status"] == "superseded" else retracted).append(r["id"])

    return BranchSideDiff(
        branch_id=branch_id,
        name=name,
        added=added,
        superseded=superseded,
        retracted=retracted,
    )


def diff(
    db: Database, actor: str, branch_a: str | uuid.UUID, branch_b: str | uuid.UUID
) -> BranchDiff:
    """Report what each branch added, superseded, and retracted (audited read)."""

    def work(conn: psycopg.Connection) -> BranchDiff:
        with conn.cursor(row_factory=dict_row) as cur:
            a = resolve_branch(cur, branch_a)
            b = resolve_branch(cur, branch_b)
            side_a = _side_diff(cur, a["id"], a["name"])
            side_b = _side_diff(cur, b["id"], b["name"])
            audit.record(
                conn,
                actor=actor,
                op="diff",
                target_type="branch",
                target_id=a["id"],
                payload={
                    "branch_a": str(a["id"]),
                    "branch_b": str(b["id"]),
                    "a_added": len(side_a.added),
                    "b_added": len(side_b.added),
                },
            )
            return BranchDiff(a=side_a, b=side_b)

    return db.run_in_transaction(work)


def get_branch(db: Database, branch: str | uuid.UUID) -> Branch:
    """Fetch a branch row by name or id."""

    def work(conn: psycopg.Connection) -> Branch:
        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            cur.execute(
                "SELECT id, name, parent_branch_id, fork_point_ts, status, "
                "created_by, created_at FROM branches WHERE id = %s",
                (b["id"],),
            )
            row = cur.fetchone()
            if row is None:
                raise NotFoundError(f"branch not found: {branch}")
            return Branch.model_validate(row)

    return db.run_in_transaction(work, read_only=True)

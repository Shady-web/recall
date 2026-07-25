"""The non-semantic core of the Recall memory kernel.

`MemoryKernel` is the single entry point for reading and writing memories,
decisions, and their provenance. It is the *only* place SQL for these entities
lives (per CONTEXT.md: the kernel is never bypassed).

Two invariants are enforced structurally here:

* **Every operation is audited in its own transaction.** Each method runs inside
  one serializable transaction (with automatic 40001 retry) and calls
  :func:`kernel.audit.record` exactly once within it. Because the audit insert
  shares the transaction, a failed operation leaves zero audit rows.

* **Writes require an actor and respect read-only mode.** A kernel cannot be
  constructed without an ``actor``, so no write can run unattributed; and when
  ``read_only`` is set every write path raises :class:`ReadOnlyError` before
  touching the database. Reads are still permitted in read-only mode (and are
  themselves audited).
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from kernel import audit
from kernel.db import Database, get_default_database
from kernel.errors import InvalidStateError, NotFoundError, ReadOnlyError
from kernel.models import Decision, Memory

# Columns selected for Memory rows. embedding is intentionally excluded in
# Phase 1 (always NULL; populated in Phase 2).
_MEMORY_COLS = (
    "id, branch_id, kind, content, source, confidence, status, "
    "superseded_by, metadata, created_at"
)
_DECISION_COLS = (
    "id, branch_id, agent_id, input_hash, action, rationale, outcome, created_at"
)


class MemoryKernel:
    """Branch-scoped memory operations backed by CockroachDB.

    Construct with an explicit ``actor`` (stamped onto every audit row) and an
    optional ``read_only`` flag. Use :meth:`from_settings` to wire both from the
    process configuration (``RECALL_ACTOR_ID`` / ``RECALL_READ_ONLY``).
    """

    def __init__(self, db: Database, *, actor: str, read_only: bool = False) -> None:
        if not actor:
            raise ValueError("MemoryKernel requires a non-empty actor")
        self.db = db
        self.actor = actor
        self.read_only = read_only

    @classmethod
    def from_settings(cls, db: Database | None = None) -> MemoryKernel:
        """Build a kernel using the process configuration."""
        from kernel.config import settings

        return cls(
            db or get_default_database(),
            actor=settings.recall_actor_id,
            read_only=settings.recall_read_only,
        )

    # -- internal helpers -------------------------------------------------

    def _require_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyError(
                "kernel is in read-only mode (RECALL_READ_ONLY); writes are blocked"
            )

    @staticmethod
    def _resolve_branch(cur: psycopg.Cursor, branch: str | uuid.UUID) -> dict[str, Any]:
        """Resolve a branch by id (if it parses as a UUID) or by name."""
        row = None
        try:
            uuid.UUID(str(branch))
        except ValueError:
            pass
        else:
            cur.execute("SELECT id, name FROM branches WHERE id = %s", (branch,))
            row = cur.fetchone()
        if row is None:
            cur.execute("SELECT id, name FROM branches WHERE name = %s", (str(branch),))
            row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"branch not found: {branch}")
        return row

    # -- write paths ------------------------------------------------------

    def remember(
        self,
        branch: str | uuid.UUID,
        content: str,
        kind: str,
        source: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Record a new memory on ``branch`` and audit it."""
        self._require_writable()

        def work(conn: psycopg.Connection) -> Memory:
            with conn.cursor(row_factory=dict_row) as cur:
                b = self._resolve_branch(cur, branch)
                cur.execute(
                    f"INSERT INTO memories "
                    f"  (branch_id, kind, content, source, confidence, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s) "
                    f"RETURNING {_MEMORY_COLS}",
                    (b["id"], kind, content, source, confidence, Jsonb(metadata or {})),
                )
                row = cur.fetchone()
                audit.record(
                    conn,
                    actor=self.actor,
                    op="remember",
                    target_type="memory",
                    target_id=row["id"],
                    payload={
                        "branch_id": str(b["id"]),
                        "kind": kind,
                        "source": source,
                        "confidence": confidence,
                    },
                )
                return Memory.model_validate(row)

        return self.db.run_in_transaction(work)

    def supersede(
        self,
        memory_id: str | uuid.UUID,
        new_content: str,
        kind: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Replace an active memory with a new one, preserving history.

        The old row is marked ``superseded`` and linked to the replacement via
        ``superseded_by``; it is never deleted. Fields not supplied are inherited
        from the old memory.
        """
        self._require_writable()

        def work(conn: psycopg.Connection) -> Memory:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"SELECT {_MEMORY_COLS} FROM memories WHERE id = %s",
                    (memory_id,),
                )
                old = cur.fetchone()
                if old is None:
                    raise NotFoundError(f"memory not found: {memory_id}")
                if old["status"] != "active":
                    raise InvalidStateError(
                        f"cannot supersede memory {memory_id} in status "
                        f"{old['status']!r}; only 'active' memories can be superseded"
                    )

                cur.execute(
                    f"INSERT INTO memories "
                    f"  (branch_id, kind, content, source, confidence, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s) "
                    f"RETURNING {_MEMORY_COLS}",
                    (
                        old["branch_id"],
                        kind if kind is not None else old["kind"],
                        new_content,
                        source if source is not None else old["source"],
                        confidence if confidence is not None else old["confidence"],
                        Jsonb(metadata if metadata is not None else old["metadata"]),
                    ),
                )
                new_row = cur.fetchone()
                cur.execute(
                    "UPDATE memories SET status = 'superseded', superseded_by = %s "
                    "WHERE id = %s",
                    (new_row["id"], memory_id),
                )
                audit.record(
                    conn,
                    actor=self.actor,
                    op="supersede",
                    target_type="memory",
                    target_id=new_row["id"],
                    payload={"superseded": str(memory_id)},
                )
                return Memory.model_validate(new_row)

        return self.db.run_in_transaction(work)

    def retract(self, memory_id: str | uuid.UUID, reason: str) -> Memory:
        """Mark a memory ``retracted`` (never deleted) and audit the reason."""
        self._require_writable()

        def work(conn: psycopg.Connection) -> Memory:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"UPDATE memories SET status = 'retracted' WHERE id = %s "
                    f"RETURNING {_MEMORY_COLS}",
                    (memory_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise NotFoundError(f"memory not found: {memory_id}")
                audit.record(
                    conn,
                    actor=self.actor,
                    op="retract",
                    target_type="memory",
                    target_id=memory_id,
                    payload={"reason": reason},
                )
                return Memory.model_validate(row)

        return self.db.run_in_transaction(work)

    def record_decision(
        self,
        branch: str | uuid.UUID,
        agent_id: str,
        action: str,
        rationale: str | None = None,
        memory_ids: list[str | uuid.UUID] | None = None,
    ) -> Decision:
        """Record a decision and its ``decision_memories`` provenance rows."""
        self._require_writable()
        memory_ids = list(memory_ids or [])

        def work(conn: psycopg.Connection) -> Decision:
            with conn.cursor(row_factory=dict_row) as cur:
                b = self._resolve_branch(cur, branch)
                cur.execute(
                    f"INSERT INTO decisions (branch_id, agent_id, action, rationale) "
                    f"VALUES (%s, %s, %s, %s) RETURNING {_DECISION_COLS}",
                    (b["id"], agent_id, action, rationale),
                )
                drow = cur.fetchone()
                for rank, mid in enumerate(memory_ids):
                    cur.execute(
                        "INSERT INTO decision_memories (decision_id, memory_id, rank) "
                        "VALUES (%s, %s, %s)",
                        (drow["id"], mid, rank),
                    )
                audit.record(
                    conn,
                    actor=self.actor,
                    op="record_decision",
                    target_type="decision",
                    target_id=drow["id"],
                    payload={
                        "agent_id": agent_id,
                        "action": action,
                        "memory_ids": [str(m) for m in memory_ids],
                    },
                )
                return Decision.model_validate(drow)

        return self.db.run_in_transaction(work)

    # -- read paths (audited, allowed in read-only mode) ------------------

    def get(self, memory_id: str | uuid.UUID) -> Memory:
        """Fetch a single memory by id (audited)."""

        def work(conn: psycopg.Connection) -> Memory:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"SELECT {_MEMORY_COLS} FROM memories WHERE id = %s",
                    (memory_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise NotFoundError(f"memory not found: {memory_id}")
                audit.record(
                    conn,
                    actor=self.actor,
                    op="get",
                    target_type="memory",
                    target_id=memory_id,
                    payload={},
                )
                return Memory.model_validate(row)

        return self.db.run_in_transaction(work)

    def list_memories(
        self,
        branch: str | uuid.UUID,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories on ``branch`` newest-first, with optional filters (audited)."""

        def work(conn: psycopg.Connection) -> list[Memory]:
            with conn.cursor(row_factory=dict_row) as cur:
                b = self._resolve_branch(cur, branch)
                query = f"SELECT {_MEMORY_COLS} FROM memories WHERE branch_id = %s"
                params: list[Any] = [b["id"]]
                if kind is not None:
                    query += " AND kind = %s"
                    params.append(kind)
                if status is not None:
                    query += " AND status = %s"
                    params.append(status)
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                cur.execute(query, params)
                rows = cur.fetchall()
                audit.record(
                    conn,
                    actor=self.actor,
                    op="list",
                    target_type="branch",
                    target_id=b["id"],
                    payload={
                        "kind": kind,
                        "status": status,
                        "limit": limit,
                        "offset": offset,
                        "count": len(rows),
                    },
                )
                return [Memory.model_validate(r) for r in rows]

        return self.db.run_in_transaction(work)

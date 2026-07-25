"""Audit logging — the project's single most important invariant.

Every kernel operation writes exactly one row to ``audit_log`` INSIDE the same
transaction as the operation itself. If the operation rolls back, its audit row
rolls back with it; an operation that commits without an audit row is a
correctness bug.

This module deliberately exposes only :func:`record`, which requires a non-empty
``actor``. Kernel write/read paths call it with the connection that is already
running their transaction, so the audit insert cannot land in a different
transaction. The ``actor`` requirement is enforced here and, structurally,
upstream: a :class:`~kernel.memory.MemoryKernel` cannot be constructed without an
actor, so no write path can run without one.
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def record(
    conn: psycopg.Connection,
    *,
    actor: str,
    op: str,
    target_type: str,
    target_id: str | uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Insert one audit row using ``conn`` (the caller's live transaction).

    Raises ``ValueError`` if ``actor`` is empty — an unattributed audit row is
    worse than useless, so we refuse to write one.
    """
    if not actor:
        raise ValueError("audit requires a non-empty actor")
    conn.execute(
        "INSERT INTO audit_log (actor, op, target_type, target_id, payload) "
        "VALUES (%s, %s, %s, %s, %s)",
        (actor, op, target_type, target_id, Jsonb(payload or {})),
    )

"""Pydantic models mirroring the Recall schema rows.

These are plain data carriers returned by the kernel API. They are built from
``dict`` rows (psycopg ``dict_row``) via :meth:`model_validate`. Columns that a
query omits fall back to the field default (e.g. ``embedding`` is not selected in
Phase 1 and defaults to ``None``).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class _Row(BaseModel):
    # from_attributes lets us validate ORM-ish objects too; extra="ignore" keeps
    # validation robust if a query returns more columns than a model declares.
    model_config = ConfigDict(from_attributes=True, extra="ignore")


class Branch(_Row):
    id: uuid.UUID
    name: str
    parent_branch_id: uuid.UUID | None = None
    fork_point_ts: datetime | None = None
    status: str
    created_by: str
    created_at: datetime


class Memory(_Row):
    id: uuid.UUID
    branch_id: uuid.UUID
    kind: str
    content: str
    # Populated in Phase 2; not selected by Phase 1 queries.
    embedding: list[float] | None = None
    source: str | None = None
    confidence: float
    status: str
    superseded_by: uuid.UUID | None = None
    metadata: dict[str, Any] = {}
    created_at: datetime


class Decision(_Row):
    id: uuid.UUID
    branch_id: uuid.UUID
    agent_id: str
    input_hash: str | None = None
    action: str
    rationale: str | None = None
    outcome: str | None = None
    created_at: datetime


class AuditEntry(_Row):
    id: uuid.UUID
    actor: str
    op: str
    target_type: str
    target_id: uuid.UUID | None = None
    payload: dict[str, Any] = {}
    created_at: datetime

"""Boundary validation for Recall MCP tool arguments.

Every tool validates its arguments through one of these models *before* any
kernel call is made. Two consequences, both deliberate:

* A bad argument never reaches the database, and never costs an embedding call.
* Failures come back as a typed ``invalid_input`` error listing the offending
  fields (see :func:`mcp_server.errors.error_payload`), rather than as an
  opaque protocol-level error.

``extra="forbid"`` is on every model: a misspelled argument is a loud failure,
not a silently ignored one. That matters most for ``filters`` — a typo'd
``min_confidance`` that was quietly dropped would return *more* memories than
the caller asked for while looking like it worked.

**Lookup vs. creation is validated asymmetrically, on purpose.** Fields that
*resolve* an existing branch (``branch``, ``parent``, ``branch_a`` …) stay
permissive, because the kernel accepts either a branch name or a UUID and it is
not this layer's job to guess which. The one field that *creates* a name in a
shared, uniquely-indexed namespace — ``branch.name`` — is held to a strict
pattern, since a name containing whitespace or control characters would be
permanent and awkward for every later caller.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Statuses a memory can hold (mirrors the CHECK constraint in migration 001).
MemoryStatus = Literal["active", "superseded", "retracted"]

#: Branch names must start alphanumeric, then allow word/dot/slash/dash. Applied
#: only when a name is being created — never when one is being looked up.
BRANCH_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$"

#: A branch reference accepted for lookup: a name or a UUID string. Bounded in
#: length so a pathological argument is never forwarded to the database.
BranchRef = Annotated[str, Field(min_length=1, max_length=256)]


class _Request(BaseModel):
    """Base for tool argument models: strict, whitespace-trimmed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RememberRequest(_Request):
    branch: BranchRef
    content: str = Field(min_length=1, max_length=100_000)
    kind: str = Field(min_length=1, max_length=64)
    source: str | None = Field(default=None, max_length=512)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RecallFilters(_Request):
    """Structured filters applied alongside vector similarity.

    ``status`` defaults to ``"active"`` to match the kernel's own default;
    passing ``null`` explicitly widens the search to every status.
    """

    kind: str | None = Field(default=None, min_length=1, max_length=64)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    since: datetime | None = None
    status: MemoryStatus | None = "active"


class RecallRequest(_Request):
    branch: BranchRef
    query: str = Field(min_length=1, max_length=10_000)
    k: int = Field(default=10, ge=1, le=100)
    filters: RecallFilters = Field(default_factory=RecallFilters)

    @field_validator("filters", mode="before")
    @classmethod
    def _omitted_filters_are_defaults(cls, value: object) -> object:
        """Treat an explicit ``null`` the same as omitting ``filters`` entirely.

        MCP clients spell "no filters" as ``null`` far more often than by leaving
        the key out, and rejecting that would be a pointless papercut.
        """
        return RecallFilters() if value is None else value


class BranchRequest(_Request):
    """Arguments for forking a branch. ``name`` is created, so it is strict."""

    parent: BranchRef
    name: str = Field(min_length=1, max_length=128, pattern=BRANCH_NAME_PATTERN)


class CommitRequest(_Request):
    branch: BranchRef


class DiscardRequest(_Request):
    branch: BranchRef
    reason: str = Field(default="", max_length=1024)


class DiffRequest(_Request):
    branch_a: BranchRef
    branch_b: BranchRef


class DecisionRequest(_Request):
    """Shared by ``explain_decision`` and ``rewind``.

    Typing this as :class:`uuid.UUID` means a malformed id is rejected here with
    a field-level message instead of reaching SQL.
    """

    decision_id: uuid.UUID

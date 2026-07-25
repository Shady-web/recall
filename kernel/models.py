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


class RecallResult(BaseModel):
    """One hit from a recall query: a memory plus its score and 1-based rank."""

    memory: Memory
    similarity: float
    rank: int


class AncestrySegment(BaseModel):
    """One link in a branch's ancestry chain.

    ``visible_as_of`` is the upper bound on ``created_at`` for memories on this
    branch to be visible from the branch the ancestry was resolved for. It is
    ``None`` for the branch itself (its own writes are always visible, live).
    """

    branch_id: uuid.UUID
    name: str
    depth: int
    visible_as_of: datetime | None = None


class MemoryConflict(BaseModel):
    """A commit-time conflict, returned as data rather than raised.

    Raised (as data) when a branch modified an inherited memory that the parent
    also changed after the fork point.
    """

    memory_id: uuid.UUID
    content: str
    branch_status: str
    parent_status: str
    parent_changed_at: datetime
    fork_point_ts: datetime
    reason: str = "memory was modified on both the branch and the parent after the fork point"


class CommitResult(BaseModel):
    """Outcome of :func:`kernel.branching.commit`.

    When ``committed`` is False the commit was a no-op: nothing was replayed and
    the branch is left open. The caller decides what to do about ``conflicts``.
    """

    branch_id: uuid.UUID
    committed: bool
    replayed_memory_ids: list[uuid.UUID] = []
    applied_override_ids: list[uuid.UUID] = []
    conflicts: list[MemoryConflict] = []


class BranchSideDiff(BaseModel):
    """What one side of a diff did, relative to the common ancestry."""

    branch_id: uuid.UUID
    name: str
    added: list[uuid.UUID] = []
    superseded: list[uuid.UUID] = []
    retracted: list[uuid.UUID] = []


class BranchDiff(BaseModel):
    """Symmetric diff between two branches."""

    a: BranchSideDiff
    b: BranchSideDiff


# --- Replay window -------------------------------------------------------


class ReplayWindow(BaseModel):
    """The range over which PHYSICAL (``AS OF SYSTEM TIME``) replay is possible.

    Bounded by CockroachDB's MVCC garbage collection. Logical replay is not
    bounded by this and works at any age.
    """

    gc_ttl_seconds: int
    earliest: datetime
    latest: datetime

    def contains(self, t: datetime) -> bool:
        return self.earliest <= t <= self.latest


# --- Decision provenance -------------------------------------------------


class ContributingMemory(BaseModel):
    """One memory that fed a decision, then vs now.

    ``similarity`` and ``rank`` are as recorded **at decision time**;
    ``status_now`` is the memory's effective status **today** on the decision's
    branch. ``invalidated`` is the headline signal: the agent acted on this, and
    we have since learned it was wrong or withdrawn.
    """

    memory_id: uuid.UUID
    content: str
    kind: str
    source: str | None = None
    confidence: float
    branch_id: uuid.UUID
    created_at: datetime

    # As recorded at decision time.
    similarity: float | None = None
    rank: int

    # As of now.
    status_now: str
    invalidated: bool = False
    superseded: bool = False
    retracted: bool = False
    superseded_by: uuid.UUID | None = None
    superseded_at: datetime | None = None
    retracted_at: datetime | None = None


class DecisionExplanation(BaseModel):
    """Full provenance for one decision: what drove it, and what has changed.

    ``has_invalidated_memories`` / ``invalidated_memory_ids`` are deliberately
    top-level so a UI can flag a suspect decision without walking the list.
    """

    decision: Decision
    branch_id: uuid.UUID
    branch_name: str
    memories: list[ContributingMemory] = []

    # Prominent flags: this decision rested on something since invalidated.
    has_invalidated_memories: bool = False
    invalidated_count: int = 0
    invalidated_memory_ids: list[uuid.UUID] = []


# --- Rewind / re-run -----------------------------------------------------


class MemoryRef(BaseModel):
    """Compact memory reference for diffs and UI lists."""

    memory_id: uuid.UUID
    content: str
    kind: str
    status: str


class MemoryAvailabilityDiff(BaseModel):
    """Which memories were available at decision time vs now."""

    only_then: list[MemoryRef] = []   # available then, gone now (retracted/superseded)
    only_now: list[MemoryRef] = []    # learned since the decision
    common_count: int = 0
    then_count: int = 0
    now_count: int = 0


class AgentDecision(BaseModel):
    """Normalized return value of an injected agent callable."""

    action: str
    rationale: str | None = None


class RerunContext(BaseModel):
    """What an injected agent callable receives on a re-run.

    ``memories`` is the branch state reconstructed at ``replayed_at`` — exactly
    what the agent could have known when the original decision was made.
    """

    decision: Decision
    memories: list[Memory] = []
    replayed_at: datetime


class RerunResult(BaseModel):
    """Old decision vs a fresh decision made against reconstructed context."""

    decision_id: uuid.UUID
    branch_id: uuid.UUID
    branch_name: str
    decision_at: datetime
    # Which reconstructed state the agent was actually run against. Defaults to
    # `decision_at` (faithful replay); set to a later time to ask "given what we
    # know now, would it still decide this?".
    evaluated_at: datetime

    old_action: str
    old_rationale: str | None = None
    new_action: str
    new_rationale: str | None = None
    action_changed: bool = False

    memory_diff: MemoryAvailabilityDiff
    contributing_memory_ids: list[uuid.UUID] = []

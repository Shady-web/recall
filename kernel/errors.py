"""Exception types raised by the Recall kernel."""

from __future__ import annotations


class RecallError(Exception):
    """Base class for all kernel errors."""


class ReadOnlyError(RecallError):
    """Raised by a write path when the kernel is in read-only mode.

    Triggered before any database work happens (see ``RECALL_READ_ONLY``).
    """


class NotFoundError(RecallError):
    """Raised when a referenced branch, memory, or decision does not exist."""


class InvalidStateError(RecallError):
    """Raised when an operation is illegal for an entity's current state.

    e.g. superseding a memory that is not ``active``.
    """


class EmbeddingError(RecallError):
    """Raised when producing an embedding fails (e.g. Bedrock error/throttling)."""


class ReplayWindowExpiredError(RecallError):
    """Raised when a PHYSICAL replay reaches past CockroachDB's MVCC GC window.

    Physical replay uses ``AS OF SYSTEM TIME``, which cannot read older than
    ``gc.ttlseconds``. Rather than returning empty or misleading data (the raw
    engine error is often something unhelpful like ``database ... does not
    exist``), the kernel refuses up front and names the window it can serve.

    Logical replay (:func:`kernel.replay.replay_branch_at`) has no such limit.
    """

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

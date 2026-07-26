"""Typed error and success envelopes for the Recall MCP server.

Every Recall tool returns **structured JSON, never prose and never a stack
trace**. Two shapes, and only two:

    {"ok": true,  "tool": "<name>", "data": { ... }}
    {"ok": false, "tool": "<name>", "error": {"type": ..., "message": ...}}

The ``type`` field is a small closed vocabulary (:class:`ErrorType`) so an MCP
client can branch on it programmatically instead of pattern-matching English.
Kernel exceptions map onto it in :func:`error_payload`; anything unmapped becomes
``internal`` and the full traceback is logged **server-side only**.

Note what is *not* an error here: commit conflicts. The kernel returns those as
data (``CommitResult.conflicts``), so the ``commit`` tool answers ``ok: true``
with ``committed: false`` and the conflicts attached. A conflict is an outcome,
not a failure.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from kernel.errors import (
    EmbeddingError,
    InvalidStateError,
    NotFoundError,
    ReadOnlyError,
    ReplayWindowExpiredError,
)

logger = logging.getLogger("recall.mcp.errors")


class ErrorType(StrEnum):
    """Closed vocabulary of machine-readable error types."""

    READ_ONLY = "read_only"
    NOT_FOUND = "not_found"
    INVALID_STATE = "invalid_state"
    INVALID_INPUT = "invalid_input"
    EMBEDDING_FAILED = "embedding_failed"
    REPLAY_WINDOW_EXPIRED = "replay_window_expired"
    INTERNAL = "internal"


class ToolInputError(Exception):
    """Raised by boundary validation before any kernel call happens.

    Carries per-field detail so the caller can fix the specific argument rather
    than guess at the whole payload.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# Exception -> (error type, is the caller able to retry the call unchanged?).
# Order matters: the first match wins, so subclasses precede their bases.
# pydantic's ValidationError is handled ahead of this table (it needs field-level
# detail), which is also why it must not be reached via the ValueError row.
_EXCEPTION_MAP: list[tuple[type[BaseException], ErrorType, bool]] = [
    (ReadOnlyError, ErrorType.READ_ONLY, False),
    (NotFoundError, ErrorType.NOT_FOUND, False),
    (InvalidStateError, ErrorType.INVALID_STATE, False),
    (ReplayWindowExpiredError, ErrorType.REPLAY_WINDOW_EXPIRED, False),
    # Bedrock throttling is the common case, and it clears on its own.
    (EmbeddingError, ErrorType.EMBEDDING_FAILED, True),
    (ToolInputError, ErrorType.INVALID_INPUT, False),
    (ValueError, ErrorType.INVALID_INPUT, False),
]


def _validation_fields(exc: ValidationError) -> list[dict[str, str]]:
    """Flatten a pydantic ValidationError into field/message pairs."""
    return [
        {
            "field": ".".join(str(p) for p in err["loc"]) or "<root>",
            "message": err["msg"],
        }
        for err in exc.errors()
    ]


def _validation_message(tool: str, fields: list[dict[str, str]]) -> str:
    """One-line summary of a validation failure.

    Pydantic's own ``str(exc)`` runs to several lines per field and ends in a
    docs URL — accurate, but noise for a model reading a tool result. The
    per-field detail is preserved in ``details.fields``.
    """
    joined = "; ".join(f"{f['field']}: {f['message']}" for f in fields)
    return f"invalid arguments for {tool!r} — {joined}"


def ok_payload(tool: str, data: dict[str, Any]) -> dict[str, Any]:
    """Wrap a successful tool result."""
    return {"ok": True, "tool": tool, "data": data}


def error_payload(tool: str, exc: BaseException) -> dict[str, Any]:
    """Map an exception onto a typed error object.

    Unmapped exceptions are logged with their traceback here and reported as
    ``internal`` carrying only the exception class and message — enough to
    diagnose, without shipping a traceback across the MCP boundary.
    """
    if isinstance(exc, ValidationError):
        fields = _validation_fields(exc)
        return build_error(
            tool,
            ErrorType.INVALID_INPUT,
            _validation_message(tool, fields),
            details={"fields": fields},
        )

    for exc_type, error_type, retryable in _EXCEPTION_MAP:
        if isinstance(exc, exc_type):
            details = exc.details if isinstance(exc, ToolInputError) else {}
            return build_error(
                tool, error_type, str(exc), details=details, retryable=retryable
            )

    logger.exception("unhandled exception in MCP tool %r", tool)
    return build_error(
        tool,
        ErrorType.INTERNAL,
        f"{type(exc).__name__}: {exc}",
        details={"exception": type(exc).__name__},
        retryable=True,
    )


def build_error(
    tool: str,
    error_type: ErrorType,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    """Construct an error envelope directly (for refusals with no exception)."""
    return {
        "ok": False,
        "tool": tool,
        "error": {
            "type": str(error_type),
            "message": message,
            "retryable": retryable,
            "details": details or {},
        },
    }

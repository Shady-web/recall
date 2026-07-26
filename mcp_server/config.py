"""Configuration for the Recall MCP server.

Reuses ``kernel.config.settings`` for everything the kernel already owns (DSN,
Bedrock, ``RECALL_READ_ONLY``) and adds only what is MCP-specific. Kernel
configuration is deliberately not extended — the MCP server is a client of the
kernel, not a peer.

**Actor identity is the point of this module.** Audit rows must distinguish an
operation initiated over MCP from the same operation made by a direct kernel
caller, so every actor this server uses is forced to carry the
:data:`MCP_ACTOR_PREFIX`. A deployment can rename the identity
(``RECALL_MCP_ACTOR``) but cannot drop the prefix — :func:`mcp_actor` re-adds it.
That makes ``SELECT * FROM audit_log WHERE actor LIKE 'mcp:%'`` a reliable
question to ask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Every audit row written on behalf of an MCP tool call carries an actor
#: starting with this. Non-negotiable — see the module docstring.
MCP_ACTOR_PREFIX = "mcp:"

DEFAULT_SERVER_NAME = "recall"


def mcp_actor(identity: str) -> str:
    """Return ``identity`` guaranteed to carry the MCP actor prefix."""
    identity = identity.strip()
    if not identity:
        raise ValueError("MCP actor identity must be non-empty")
    if identity.startswith(MCP_ACTOR_PREFIX):
        return identity
    return f"{MCP_ACTOR_PREFIX}{identity}"


@dataclass(frozen=True)
class McpSettings:
    """Resolved runtime configuration for one MCP server process."""

    actor: str
    read_only: bool
    server_name: str = DEFAULT_SERVER_NAME


def load_mcp_settings() -> McpSettings:
    """Build settings from the process environment.

    ``RECALL_MCP_ACTOR`` overrides the identity (default: ``RECALL_ACTOR_ID``);
    ``RECALL_READ_ONLY`` is read from the kernel settings so the MCP server and
    direct kernel callers can never disagree about it.
    """
    from kernel.config import settings

    identity = os.environ.get("RECALL_MCP_ACTOR") or settings.recall_actor_id
    return McpSettings(
        actor=mcp_actor(identity),
        read_only=settings.recall_read_only,
        server_name=os.environ.get("RECALL_MCP_SERVER_NAME", DEFAULT_SERVER_NAME),
    )

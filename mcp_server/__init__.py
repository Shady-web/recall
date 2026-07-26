"""Recall MCP server — the kernel, exposed to Claude Code, Cursor, and VS Code.

A thin Model Context Protocol wrapper over :mod:`kernel`, built on the official
Python MCP SDK. It holds no product logic and never talks SQL: every tool
validates its arguments, then calls exactly one kernel entry point, then
serializes the result to JSON.

Run it over stdio with ``python -m mcp_server``. See :mod:`mcp_server.server`
for the tool list and the read-only and audit-identity guarantees.

This is *our* server, distinct from the managed CockroachDB Cloud MCP server
used during development to introspect the live cluster.
"""

from mcp_server.config import MCP_ACTOR_PREFIX, McpSettings, load_mcp_settings
from mcp_server.errors import ErrorType
from mcp_server.server import (
    READ_TOOLS,
    WRITE_TOOLS,
    build_server,
    create_server,
    main,
)

__all__ = [
    "MCP_ACTOR_PREFIX",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "ErrorType",
    "McpSettings",
    "build_server",
    "create_server",
    "load_mcp_settings",
    "main",
]

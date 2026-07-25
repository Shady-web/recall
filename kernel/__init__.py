"""Recall memory kernel.

This package is the *only* component permitted to talk SQL to CockroachDB.
Every other component (MCP server, agent, UI backend) must go through the
kernel. If a component outside this package writes SQL directly, that is a bug.
"""

__all__ = [
    "audit",
    "backfill",
    "config",
    "db",
    "embeddings",
    "errors",
    "memory",
    "migrate",
    "models",
    "recall",
]

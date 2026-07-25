"""Minimal forward-only migration runner for Recall.

Applies numbered ``NNN_*.sql`` files from the ``migrations/`` directory in order,
recording each applied version in a ``schema_migrations`` ledger so the runner is
idempotent and safe to re-run: already-applied migrations are skipped.

Why no single wrapping transaction per file: CockroachDB (v25.2) auto-commits
before processing DDL (``autocommit_before_ddl``), so a file mixing DDL and DML
cannot be applied as one atomic unit. We compensate two ways: (1) migration files
use idempotent DDL (``IF NOT EXISTS`` / ``ON CONFLICT``) so a partially-applied
file re-applies cleanly, and (2) the ledger row is only written after the file
runs, so an interrupted migration is retried on the next run.

Usage:
    python -m kernel.migrate                 # uses CRDB_CONNECTION_STRING
    python -m kernel.migrate --dsn <dsn>     # explicit target
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import psycopg

logger = logging.getLogger("recall.kernel.migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Matches "001_init.sql" -> version "001".
_VERSION_RE = re.compile(r"^(\d+)_.*\.sql$")

_CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    STRING PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _dsn_from_settings() -> str:
    # Imported lazily so importing this module never requires env configuration.
    from kernel.config import settings

    return settings.crdb_connection_string


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[tuple[str, Path]]:
    """Return ``(version, path)`` pairs for migration files, ordered by version."""
    found: list[tuple[str, Path]] = []
    for path in migrations_dir.glob("*.sql"):
        match = _VERSION_RE.match(path.name)
        if match:
            found.append((match.group(1), path))
    found.sort(key=lambda vp: vp[0])
    return found


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    conn.execute(_CREATE_LEDGER)
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def migrate(
    dsn: str | None = None,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[str]:
    """Apply all pending migrations and return the versions newly applied.

    Running this a second time against the same database is a no-op and returns
    an empty list.
    """
    dsn = dsn or _dsn_from_settings()
    newly_applied: list[str] = []

    # autocommit=True: each statement commits as it runs, which is required given
    # CockroachDB's autocommit-before-DDL behaviour.
    with psycopg.connect(dsn, autocommit=True) as conn:
        already = _applied_versions(conn)
        for version, path in discover_migrations(migrations_dir):
            if version in already:
                logger.info("skip %s (already applied)", path.name)
                continue
            logger.info("applying %s", path.name)
            conn.execute(path.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING",
                (version,),
            )
            newly_applied.append(version)

    return newly_applied


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Apply Recall database migrations.")
    parser.add_argument(
        "--dsn",
        default=None,
        help="CockroachDB connection string (defaults to CRDB_CONNECTION_STRING).",
    )
    args = parser.parse_args()

    applied = migrate(dsn=args.dsn)
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    else:
        print("No pending migrations; database is up to date.")


if __name__ == "__main__":
    main()

"""Migration runner tests against a live CockroachDB instance."""

from __future__ import annotations

import psycopg

from kernel.migrate import discover_migrations, migrate
from tests.conftest import requires_crdb

pytestmark = requires_crdb

_ALL_VERSIONS = [version for version, _ in discover_migrations()]


def test_migrate_applies_then_is_idempotent(empty_dsn: str):
    first = migrate(dsn=empty_dsn)
    assert first == _ALL_VERSIONS  # every migration ran, in order

    second = migrate(dsn=empty_dsn)
    assert second == []  # nothing left to do the second time


def test_migrate_records_versions_once(empty_dsn: str):
    migrate(dsn=empty_dsn)
    migrate(dsn=empty_dsn)  # re-run
    with psycopg.connect(empty_dsn) as conn:
        rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    versions = [r[0] for r in rows]
    # Each migration recorded exactly once, no duplicates on re-run.
    assert versions == _ALL_VERSIONS


def test_seed_root_branch(test_dsn: str):
    with psycopg.connect(test_dsn) as conn:
        rows = conn.execute(
            "SELECT name, parent_branch_id, fork_point_ts, status "
            "FROM branches WHERE name = 'main'"
        ).fetchall()
    assert len(rows) == 1
    name, parent, fork_ts, status = rows[0]
    assert name == "main"
    assert parent is None
    assert fork_ts is None
    assert status == "open"


def test_seed_root_branch_not_duplicated_on_rerun(empty_dsn: str):
    migrate(dsn=empty_dsn)
    migrate(dsn=empty_dsn)
    with psycopg.connect(empty_dsn) as conn:
        count = conn.execute(
            "SELECT count(*) FROM branches WHERE name = 'main'"
        ).fetchone()[0]
    assert count == 1


def test_embedding_column_is_vector_1024(test_dsn: str):
    # A 1024-dim vector inserts fine; a wrong dimension is rejected — proving the
    # column really is VECTOR(1024).
    with psycopg.connect(test_dsn) as conn:
        branch_id = conn.execute(
            "SELECT id FROM branches WHERE name = 'main'"
        ).fetchone()[0]
        vec = "[" + ",".join(["0.1"] * 1024) + "]"
        conn.execute(
            "INSERT INTO memories (branch_id, kind, content, embedding) "
            "VALUES (%s, 'fact', 'has embedding', %s)",
            (branch_id, vec),
        )
        conn.commit()
        bad = "[" + ",".join(["0.1"] * 512) + "]"
        try:
            conn.execute(
                "INSERT INTO memories (branch_id, kind, content, embedding) "
                "VALUES (%s, 'fact', 'bad dim', %s)",
                (branch_id, bad),
            )
            conn.commit()
            raised = False
        except psycopg.Error:
            conn.rollback()
            raised = True
    assert raised, "a 512-dim vector should be rejected by a VECTOR(1024) column"

"""Pytest fixtures for tests that run against a live CockroachDB instance.

Set ``RECALL_TEST_DSN`` to point at a running cluster; it defaults to a local
insecure single-node instance:

    docker run -d --name recall-crdb -p 26257:26257 \\
        cockroachdb/cockroach:latest-v25.2 start-single-node --insecure

Each test that needs a database gets its own freshly-created, migrated database,
which is dropped on teardown — so tests are isolated and never see each other's
rows.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from kernel.db import Database
from kernel.embeddings import FakeEmbeddingProvider
from kernel.memory import MemoryKernel
from kernel.migrate import migrate

BASE_DSN = os.environ.get(
    "RECALL_TEST_DSN",
    "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
)


def _with_db(dsn: str, dbname: str) -> str:
    """Return ``dsn`` with its database path replaced by ``dbname``."""
    return urlunparse(urlparse(dsn)._replace(path=f"/{dbname}"))


def _admin_dsn() -> str:
    return _with_db(BASE_DSN, "defaultdb")


def _crdb_available() -> bool:
    try:
        with psycopg.connect(_admin_dsn(), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# Skip the whole suite (with a clear reason) if no cluster is reachable, rather
# than erroring out — this keeps `pytest` runnable without a database present.
requires_crdb = pytest.mark.skipif(
    not _crdb_available(),
    reason=(
        "no CockroachDB reachable at RECALL_TEST_DSN; start one with "
        "`docker run -p 26257:26257 cockroachdb/cockroach:latest-v25.2 "
        "start-single-node --insecure`"
    ),
)


@pytest.fixture(scope="session", autouse=True)
def _enable_vector_index() -> None:
    """Enable the cluster setting required by migration 002's vector index.

    This is a cluster-wide (not per-database) setting; enabling it once for the
    session is enough. A no-op if the cluster is unreachable — the DB-backed
    tests skip in that case anyway.
    """
    if not _crdb_available():
        return
    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")


@pytest.fixture
def empty_dsn() -> Iterator[str]:
    """Create a fresh, empty (un-migrated) database; drop it on teardown."""
    dbname = f"recall_test_{uuid.uuid4().hex[:12]}"
    admin = _admin_dsn()
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {dbname}")
    try:
        yield _with_db(BASE_DSN, dbname)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {dbname} CASCADE")


@pytest.fixture
def test_dsn(empty_dsn: str) -> str:
    """A fresh database with migrations applied."""
    migrate(dsn=empty_dsn)
    return empty_dsn


@pytest.fixture
def db(test_dsn: str) -> Iterator[Database]:
    database = Database(test_dsn, min_size=1, max_size=4)
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def kernel(db: Database, fake_embedder: FakeEmbeddingProvider) -> MemoryKernel:
    return MemoryKernel(db, actor="tester", read_only=False, embedder=fake_embedder)


def audit_count(dsn: str, op: str | None = None) -> int:
    """Count audit rows in ``dsn``'s database, optionally filtered by ``op``."""
    with psycopg.connect(dsn) as conn:
        if op is None:
            return conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
        return conn.execute(
            "SELECT count(*) FROM audit_log WHERE op = %s", (op,)
        ).fetchone()[0]


def table_count(dsn: str, table: str) -> int:
    with psycopg.connect(dsn) as conn:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

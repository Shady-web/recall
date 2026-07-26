"""Pytest fixtures for tests that run against a live CockroachDB instance.

The default target is a local insecure single-node instance, which
``./scripts/dev_db.sh up`` starts for you — so the usual workflow needs no
environment variable at all. Set ``RECALL_TEST_DSN`` to point somewhere else
(e.g. the cloud cluster) when you specifically want to test against it; expect
that to be roughly 40x slower per test. See DEV_SETUP.md.

Each test that needs a database gets its own freshly-created, migrated database,
which is dropped on teardown — so tests are isolated and never see each other's
rows. That isolation is why per-test database setup dominates the runtime.
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


# Generous on purpose. A local insecure node answers in milliseconds, but a
# CockroachDB Cloud cluster over TLS from a laptop routinely takes 5-6s to
# connect — and because an unreachable cluster *skips* the suite rather than
# failing it, too tight a probe turns every database-backed test green without
# running any of them. Waiting is the cheaper mistake.
_PROBE_TIMEOUT_SECONDS = 15


def _crdb_available() -> bool:
    try:
        with psycopg.connect(
            _admin_dsn(), connect_timeout=_PROBE_TIMEOUT_SECONDS
        ) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# Probed once per session rather than per use — the cloud probe costs seconds.
_CRDB_AVAILABLE = _crdb_available()

# Host:port/db of the target, safe to print: everything before the last '@' (the
# credentials) is dropped, so this never leaks a password into a skip message.
_TARGET = BASE_DSN.split("@")[-1].split("?")[0]

# Skip the database-backed tests (with a clear reason) when no cluster is
# reachable, rather than erroring out — this keeps `pytest` runnable with no
# database present.
#
# The cost of that convenience is that a misconfigured connection makes the suite
# exit 0 having run almost nothing. Set RECALL_REQUIRE_CRDB=1 to buy it back: the
# skip becomes a hard collection error. CI sets it, so a green CI run cannot mean
# "skipped everything" — it can only mean the suite actually executed.
_REQUIRE_CRDB = os.environ.get("RECALL_REQUIRE_CRDB") == "1"

if _REQUIRE_CRDB and not _CRDB_AVAILABLE:
    raise RuntimeError(
        f"RECALL_REQUIRE_CRDB=1 but no CockroachDB is reachable at {_TARGET}. "
        f"Refusing to run, because silently skipping the database-backed tests "
        f"would report success for a suite that never ran. Start a cluster with "
        f"`./scripts/dev_db.sh up`, or unset RECALL_REQUIRE_CRDB to allow skips."
    )

requires_crdb = pytest.mark.skipif(
    not _CRDB_AVAILABLE,
    reason=(
        f"no CockroachDB reachable at {_TARGET} (RECALL_TEST_DSN); "
        f"start a local one with `./scripts/dev_db.sh up`"
    ),
)


@pytest.fixture(scope="session", autouse=True)
def _enable_vector_index() -> None:
    """Enable the cluster setting required by migration 002's vector index.

    This is a cluster-wide (not per-database) setting; enabling it once for the
    session is enough. A no-op if the cluster is unreachable — the DB-backed
    tests skip in that case anyway.

    ``scripts/dev_db.sh up`` sets this too, so a developer or CI runner that
    started the cluster that way is already covered; this keeps the suite
    self-sufficient against a cluster started any other way.
    """
    if not _CRDB_AVAILABLE:
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

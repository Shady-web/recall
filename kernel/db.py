"""CockroachDB connection management for the Recall kernel.

This module owns the connection pool and the transactional entry points that the
rest of the kernel builds on. Two things here are load-bearing for correctness:

* **Serializable by default.** Every transaction runs at ``SERIALIZABLE``
  isolation. CockroachDB uses this isolation level by default; we also set it
  explicitly so the guarantee is visible in the code and independent of cluster
  defaults.
* **Automatic retry on serialization failures.** Under serializable isolation a
  transaction may be aborted with SQLSTATE ``40001`` and must be retried by the
  client. :func:`run_with_retry` implements bounded exponential backoff for
  exactly this case.

The retry helper is deliberately decoupled from the pool so it can be unit
tested against a fake connection with no database involved. ``kernel.config`` is
imported lazily (inside functions) so that importing this module never forces
environment configuration to be present.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg
from psycopg import IsolationLevel
from psycopg_pool import ConnectionPool

logger = logging.getLogger("recall.kernel.db")

#: SQLSTATE emitted by CockroachDB/Postgres when a serializable transaction is
#: aborted and the client is expected to retry the whole transaction.
SERIALIZATION_FAILURE_SQLSTATE = "40001"

#: Maximum number of attempts (initial try + retries) for a transaction that
#: keeps hitting serialization failures.
MAX_ATTEMPTS = 5

#: Base delay, in seconds, for exponential backoff between retries.
BASE_BACKOFF_SECONDS = 0.05


def is_serialization_failure(exc: BaseException) -> bool:
    """Return True if ``exc`` represents a retryable serialization failure.

    We check the SQLSTATE (``pgcode``/``sqlstate``) rather than relying solely on
    the exception class, so that a lightweight fake exception carrying
    ``sqlstate == "40001"`` is recognised in tests just like a real
    :class:`psycopg.errors.SerializationFailure` would be in production.
    """
    if isinstance(exc, psycopg.errors.SerializationFailure):
        return True
    return getattr(exc, "sqlstate", None) == SERIALIZATION_FAILURE_SQLSTATE


def _backoff_delay(attempt: int, base: float = BASE_BACKOFF_SECONDS) -> float:
    """Exponential backoff delay for a given 1-indexed attempt number."""
    return base * (2 ** (attempt - 1))


def run_with_retry[T](
    work: Callable[[], T],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_backoff: float = BASE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``work`` retrying on serialization failures with backoff.

    ``work`` should be an idempotent unit of work that performs a full
    transaction (begin → statements → commit). If it raises a serialization
    failure (SQLSTATE ``40001``) it is retried up to ``max_attempts`` times with
    exponential backoff between attempts. Any other exception propagates
    immediately, and a serialization failure on the final attempt is re-raised.

    ``sleep`` is injectable so tests can run without real delays.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return work()
        except Exception as exc:
            if is_serialization_failure(exc) and attempt < max_attempts:
                delay = _backoff_delay(attempt, base_backoff)
                logger.warning(
                    "serialization failure (40001) on attempt %d/%d; "
                    "retrying in %.3fs",
                    attempt,
                    max_attempts,
                    delay,
                )
                sleep(delay)
                continue
            raise


class Database:
    """Owns a psycopg3 connection pool and hands out serializable transactions.

    A single :class:`Database` instance is intended to be shared for the lifetime
    of the process. Construct it once (see :func:`get_default_database`) and reuse
    it; the underlying pool is safe for concurrent use.
    """

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        min_size: int = 1,
        max_size: int = 10,
        open: bool = True,
    ) -> None:
        if conninfo is None:
            # Imported lazily so that importing this module does not require the
            # environment to be configured (e.g. when unit-testing retry logic).
            from kernel.config import settings

            conninfo = settings.crdb_connection_string
        self.pool = ConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            open=open,
            kwargs={"application_name": "recall-kernel"},
        )

    @contextmanager
    def transaction(
        self,
        *,
        isolation: IsolationLevel = IsolationLevel.SERIALIZABLE,
        read_only: bool = False,
    ) -> Iterator[psycopg.Connection]:
        """Yield a connection inside a transaction at the given isolation level.

        The transaction is committed on clean exit and rolled back if the body
        raises. Note this does *not* retry on serialization failure on its own;
        wrap the call in :meth:`run_in_transaction` (or :func:`run_with_retry`)
        when you need automatic retries.
        """
        with self.pool.connection() as conn:
            conn.isolation_level = isolation
            conn.read_only = read_only
            with conn.transaction():
                yield conn

    def run_in_transaction[T](
        self,
        fn: Callable[[psycopg.Connection], T],
        *,
        isolation: IsolationLevel = IsolationLevel.SERIALIZABLE,
        read_only: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> T:
        """Run ``fn`` inside a serializable transaction, retrying on 40001.

        ``fn`` receives the live connection and may execute any number of
        statements; the whole transaction is retried as a unit if CockroachDB
        aborts it with a serialization failure.
        """

        def work() -> T:
            with self.transaction(isolation=isolation, read_only=read_only) as conn:
                return fn(conn)

        return run_with_retry(work, max_attempts=max_attempts)

    def health_check(self) -> bool:
        """Return True if a trivial ``SELECT 1`` succeeds against the cluster."""
        try:
            with self.transaction(read_only=True) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("health check failed")
            return False

    def close(self) -> None:
        """Close the underlying connection pool."""
        self.pool.close()


#: The column whose declared width every embedding must match.
EMBEDDING_TABLE = "memories"
EMBEDDING_COLUMN = "embedding"

#: CockroachDB reports the column as ``VECTOR(1024)`` in ``crdb_sql_type``;
#: ``data_type`` is only the unparameterised ``vector``, so the width comes from
#: the CockroachDB-specific column.
_VECTOR_WIDTH_RE = re.compile(r"vector\s*\(\s*(\d+)\s*\)", re.IGNORECASE)


def schema_vector_dimension(
    db: Database,
    *,
    table: str = EMBEDDING_TABLE,
    column: str = EMBEDDING_COLUMN,
) -> int | None:
    """Return the declared width of the schema's embedding column.

    Returns ``None`` when the column does not exist — an unmigrated database is
    a different problem with its own error, and reporting it as a dimension
    mismatch would be misleading.
    """
    with db.transaction(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT crdb_sql_type
              FROM information_schema.columns
             WHERE table_name = %s AND column_name = %s
            """,
            (table, column),
        ).fetchone()
    if row is None:
        return None
    match = _VECTOR_WIDTH_RE.search(row[0] or "")
    if match is None:
        return None
    return int(match.group(1))


def verify_embedding_dimension(db: Database, embedder: object) -> int | None:
    """Fail loudly if ``embedder`` and the schema disagree on vector width.

    Called at startup, where a mismatch is cheap to diagnose. Without it the
    first ``remember()`` fails inside an INSERT with a database-level complaint
    about vector width — which points at the row being written rather than at
    the real cause, a provider configured for a different number of dimensions
    than the schema was migrated for.

    ``embedder`` is anything exposing ``dimensions`` (the ``EmbeddingProvider``
    protocol); it is typed loosely so this module stays free of an import from
    :mod:`kernel.embeddings`. Returns the schema width, or ``None`` when the
    check could not run (no embedder, or an unmigrated database).
    """
    from kernel.errors import SchemaMismatchError

    provider_dim = getattr(embedder, "dimensions", None)
    if not isinstance(provider_dim, int):
        return None

    schema_dim = schema_vector_dimension(db)
    if schema_dim is None:
        logger.debug(
            "no %s.%s vector column found; skipping dimension check",
            EMBEDDING_TABLE,
            EMBEDDING_COLUMN,
        )
        return None

    if provider_dim != schema_dim:
        raise SchemaMismatchError(
            f"embedding dimension mismatch: provider "
            f"{type(embedder).__name__} produces {provider_dim}-dimension "
            f"vectors, but {EMBEDDING_TABLE}.{EMBEDDING_COLUMN} is declared "
            f"VECTOR({schema_dim}). Every write would be rejected by the "
            f"database. Point the provider at a {schema_dim}-dimension model "
            f"(or re-migrate the schema to VECTOR({provider_dim}) and rebuild "
            f"the vector index)."
        )
    logger.debug("embedding dimension check passed (%d)", schema_dim)
    return schema_dim


def stored_embedding_spaces(db: Database) -> dict[str | None, int]:
    """Return the embedding spaces present in ``memories``, with row counts.

    ``None`` counts rows written before migration 004, whose provenance is
    genuinely unknown.
    """
    with db.transaction(read_only=True) as conn:
        rows = conn.execute(
            "SELECT embedding_model, count(*) FROM memories "
            "WHERE embedding IS NOT NULL GROUP BY embedding_model"
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def verify_embedding_provider(db: Database, embedder: object) -> str | None:
    """Fail loudly if the configured provider differs from what is stored.

    :func:`verify_embedding_dimension` checks that vectors are the right
    *width*. This checks that they are from the right *space*, which is the
    invariant that actually governs whether a similarity score means anything.
    Width cannot stand in for it: Recall's Titan and fake providers both emit
    1024-dimension unit vectors and are mutually meaningless.

    Without this check the failure is silent. Querying a fake-seeded database
    with Titan returns a full page of hits whose scores hover around zero —
    cosine between unrelated spaces — with no error anywhere. Observed in this
    project; see migrations/004_embedding_provenance.sql.

    Returns the configured space id, or ``None`` when the check could not run
    (no provider, unmigrated schema, or an empty corpus).
    """
    from kernel.errors import SchemaMismatchError

    configured = getattr(embedder, "space_id", None)
    if not isinstance(configured, str):
        return None

    try:
        spaces = stored_embedding_spaces(db)
    except Exception:
        # Pre-004 schema: the column does not exist. Nothing to verify.
        logger.debug("no memories.embedding_model column; skipping provider check")
        return None

    known = {space: count for space, count in spaces.items() if space is not None}
    unknown = spaces.get(None, 0)

    if unknown:
        logger.warning(
            "%d memory row(s) predate embedding provenance (migration 004) and "
            "cannot be checked against the configured provider %r. If they were "
            "written by a different provider, their similarity scores against "
            "this one are meaningless.",
            unknown,
            configured,
        )

    foreign = {space: count for space, count in known.items() if space != configured}
    if not foreign:
        return configured

    detail = ", ".join(f"{space} ({count} rows)" for space, count in sorted(foreign.items()))
    raise SchemaMismatchError(
        f"embedding space mismatch: this process is configured for {configured!r}, "
        f"but the database already holds vectors from {detail}. Vectors from "
        f"different providers are not comparable — recall would return "
        f"confident-looking hits whose similarity scores are near-zero noise, "
        f"with no error to warn you. Either run with the provider that wrote "
        f"those rows, or re-seed this database with the configured one "
        f"(e.g. scripts/seed_incidents.py --reset). This is the check that "
        f"distinguishes 'the demo is cheap' from 'the demo is wrong'."
    )


_default_database: Database | None = None


def get_default_database() -> Database:
    """Return a lazily-constructed process-wide :class:`Database` singleton."""
    global _default_database
    if _default_database is None:
        _default_database = Database()
    return _default_database


def health_check() -> bool:
    """Convenience wrapper: health-check the default database."""
    return get_default_database().health_check()

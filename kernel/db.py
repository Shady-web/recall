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

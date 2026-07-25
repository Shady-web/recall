"""Unit tests for the serialization-failure retry logic in ``kernel.db``.

These tests use a fake connection that raises SQLSTATE 40001 a controlled number
of times before succeeding, so no real database is required.
"""

from __future__ import annotations

import pytest

from kernel.db import (
    BASE_BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    is_serialization_failure,
    run_with_retry,
)


class FakeSerializationError(Exception):
    """Stand-in for psycopg's SerializationFailure, carrying SQLSTATE 40001."""

    def __init__(self, message: str = "restart transaction: TransactionRetryError"):
        super().__init__(message)
        # ``run_with_retry`` inspects ``sqlstate`` to decide retryability.
        self.sqlstate = "40001"


class FakeConnection:
    """Fake connection whose ``execute`` fails ``fail_times`` then succeeds."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.attempts = 0

    def execute(self, sql: str) -> str:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise FakeSerializationError()
        return "ok"


def test_retries_twice_then_succeeds():
    """40001 raised twice, then success on the third attempt."""
    conn = FakeConnection(fail_times=2)
    sleeps: list[float] = []

    result = run_with_retry(lambda: conn.execute("SELECT 1"), sleep=sleeps.append)

    assert result == "ok"
    assert conn.attempts == 3  # two failures + one success
    # Slept once before each retry: exponential backoff 1x, 2x base.
    assert sleeps == [BASE_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * 2]


def test_gives_up_after_max_attempts():
    """Persistent 40001 exhausts attempts and re-raises the last failure."""
    conn = FakeConnection(fail_times=MAX_ATTEMPTS + 5)
    sleeps: list[float] = []

    with pytest.raises(FakeSerializationError):
        run_with_retry(lambda: conn.execute("SELECT 1"), sleep=sleeps.append)

    assert conn.attempts == MAX_ATTEMPTS
    # One sleep between each attempt, so MAX_ATTEMPTS - 1 sleeps total.
    assert len(sleeps) == MAX_ATTEMPTS - 1


def test_non_serialization_error_propagates_immediately():
    """A non-40001 error is not retried and no backoff sleep happens."""
    sleeps: list[float] = []

    def work():
        raise ValueError("not a serialization failure")

    with pytest.raises(ValueError):
        run_with_retry(work, sleep=sleeps.append)

    assert sleeps == []


def test_success_on_first_attempt_does_not_sleep():
    conn = FakeConnection(fail_times=0)
    sleeps: list[float] = []

    result = run_with_retry(lambda: conn.execute("SELECT 1"), sleep=sleeps.append)

    assert result == "ok"
    assert conn.attempts == 1
    assert sleeps == []


def test_respects_custom_max_attempts():
    conn = FakeConnection(fail_times=1)
    sleeps: list[float] = []

    # With max_attempts=1 there are no retries, so a single failure propagates.
    with pytest.raises(FakeSerializationError):
        run_with_retry(
            lambda: conn.execute("SELECT 1"),
            max_attempts=1,
            sleep=sleeps.append,
        )

    assert conn.attempts == 1
    assert sleeps == []


def test_is_serialization_failure_detects_sqlstate():
    assert is_serialization_failure(FakeSerializationError()) is True
    assert is_serialization_failure(ValueError("nope")) is False

    tagged = RuntimeError("boom")
    tagged.sqlstate = "40001"  # type: ignore[attr-defined]
    assert is_serialization_failure(tagged) is True

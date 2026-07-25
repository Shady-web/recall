"""Concurrency tests with REAL threads against a REAL CockroachDB cluster.

These are the tests that exercise the CockroachDB story directly: serializable
isolation, transaction retries, and the audit invariant holding under contention.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import psycopg

from kernel.branching import commit, fork, get_branch
from kernel.db import Database
from kernel.embeddings import FakeEmbeddingProvider
from kernel.memory import MemoryKernel
from tests.conftest import audit_count, requires_crdb, table_count

pytestmark = requires_crdb


def _kernel(dsn: str, actor: str) -> tuple[Database, MemoryKernel]:
    db = Database(dsn, min_size=1, max_size=8)
    return db, MemoryKernel(db, actor=actor, read_only=False, embedder=FakeEmbeddingProvider())


def test_two_agents_forking_same_parent_both_succeed(test_dsn):
    """Concurrent forks of one parent must both succeed with distinct ids."""
    db, _ = _kernel(test_dsn, "forker")
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def do_fork(name: str):
        try:
            barrier.wait(timeout=30)  # maximize the overlap
            results.append(fork(db, f"agent-{name}", "main", name))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=do_fork, args=(n,)) for n in ("alpha", "beta")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    try:
        assert errors == [], f"concurrent forks raised: {errors}"
        assert len(results) == 2
        assert results[0].id != results[1].id
        assert {r.name for r in results} == {"alpha", "beta"}
        # Both are children of main, and both forks are audited.
        main = get_branch(db, "main")
        assert all(r.parent_branch_id == main.id for r in results)
        assert audit_count(test_dsn, op="fork") == 2
    finally:
        db.close()


def test_conflicting_commits_exactly_one_wins_cleanly(test_dsn):
    """Two branches modify the same parent memory, then commit concurrently.

    Exactly one must commit cleanly; the other must come back with a structured
    conflict and leave no partial state behind.
    """
    db, kernel = _kernel(test_dsn, "setup")
    try:
        base = kernel.remember("main", "the retry limit is 3 on the server", kind="fact")
        fork(db, "agent-left", "main", "left")
        fork(db, "agent-right", "main", "right")
        kernel.supersede(base.id, "the retry limit is 5 on the server", branch="left")
        kernel.supersede(base.id, "the retry limit is 9 on the server", branch="right")

        main_before = table_count(test_dsn, "memories")
        barrier = threading.Barrier(2)
        outcomes: list = []
        errors: list = []

        def do_commit(branch: str):
            try:
                barrier.wait(timeout=30)
                outcomes.append((branch, commit(db, f"agent-{branch}", branch)))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append((branch, exc))

        threads = [threading.Thread(target=do_commit, args=(b,)) for b in ("left", "right")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"commits raised instead of returning conflicts: {errors}"
        assert len(outcomes) == 2

        winners = [(b, r) for b, r in outcomes if r.committed]
        losers = [(b, r) for b, r in outcomes if not r.committed]
        assert len(winners) == 1, f"expected exactly one winner, got {outcomes}"
        assert len(losers) == 1

        # The loser reports a structured conflict, not an exception.
        loser_branch, loser_result = losers[0]
        assert loser_result.conflicts, "loser must report a structured conflict"
        assert loser_result.conflicts[0].memory_id == base.id
        assert loser_result.replayed_memory_ids == []

        # No partial state: loser stays open, winner is committed.
        winner_branch, winner_result = winners[0]
        assert get_branch(db, winner_branch).status == "committed"
        assert get_branch(db, loser_branch).status == "open"

        # Only the winner's override was applied to the parent.
        with psycopg.connect(test_dsn) as conn:
            status = conn.execute(
                "SELECT status FROM memories WHERE id = %s", (base.id,)
            ).fetchone()[0]
        assert status == "superseded"
        # The parent grew by exactly the winner's replayed rows and nothing else:
        # the loser left no partial state behind.
        assert table_count(test_dsn, "memories") == main_before + len(
            winner_result.replayed_memory_ids
        )
    finally:
        db.close()


def test_n_concurrent_writers_no_lost_updates_and_audit_matches(test_dsn):
    """N writers on one branch: every write lands, and audit rows == successes."""
    n_writers, per_writer = 8, 6
    total = n_writers * per_writer

    db, _ = _kernel(test_dsn, "unused")
    try:
        barrier = threading.Barrier(n_writers)

        def writer(idx: int) -> int:
            k = MemoryKernel(
                db, actor=f"writer-{idx}", read_only=False, embedder=FakeEmbeddingProvider()
            )
            barrier.wait(timeout=60)
            ok = 0
            for j in range(per_writer):
                k.remember("main", f"writer {idx} memory {j} about servers", kind="fact")
                ok += 1
            return ok

        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            successes = sum(f.result(timeout=120) for f in
                            [pool.submit(writer, i) for i in range(n_writers)])

        assert successes == total
        # No lost updates: every successful write produced exactly one row...
        assert table_count(test_dsn, "memories") == total
        # ...and exactly one audit row. This is the project's core invariant,
        # holding under real contention with serialization retries.
        assert audit_count(test_dsn, op="remember") == total
        assert audit_count(test_dsn) == total

        # Every writer is attributed in the audit log.
        with psycopg.connect(test_dsn) as conn:
            actors = conn.execute(
                "SELECT DISTINCT actor FROM audit_log WHERE op = 'remember'"
            ).fetchall()
        assert len(actors) == n_writers
    finally:
        db.close()

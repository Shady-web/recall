"""Branching semantics against a live CockroachDB instance."""

from __future__ import annotations

import psycopg
import pytest

from kernel.branching import commit, diff, discard, fork, get_branch, resolve_ancestry
from kernel.errors import InvalidStateError
from tests.conftest import audit_count, requires_crdb

pytestmark = requires_crdb


# -- fork -----------------------------------------------------------------

def test_fork_sets_parent_and_fork_point(db, kernel):
    child = fork(db, "tester", "main", "experiment")
    assert child.name == "experiment"
    assert child.status == "open"
    assert child.fork_point_ts is not None
    main = get_branch(db, "main")
    assert child.parent_branch_id == main.id


def test_fork_is_audited_as_first_class_op(db, test_dsn):
    fork(db, "tester", "main", "experiment")
    assert audit_count(test_dsn, op="fork") == 1


def test_resolve_ancestry_chain_and_bounds(db, kernel):
    fork(db, "tester", "main", "child")
    grand = fork(db, "tester", "child", "grand")

    chain = resolve_ancestry(db, grand.id)
    assert [s.name for s in chain] == ["grand", "child", "main"]
    assert [s.depth for s in chain] == [0, 1, 2]
    # The branch itself is live (unbounded); ancestors are bounded and the bound
    # decreases monotonically toward the root.
    assert chain[0].visible_as_of is None
    assert chain[1].visible_as_of is not None
    assert chain[2].visible_as_of <= chain[1].visible_as_of


# -- inheritance and isolation -------------------------------------------

def test_branch_inherits_parent_memories(db, kernel):
    kernel.remember("main", "the database server crashed", kind="fact")
    fork(db, "tester", "main", "child")
    hits = kernel.recall("child", "database server", k=10)
    assert any("database server crashed" in h.memory.content for h in hits)


def test_writes_on_branch_do_not_affect_parent(db, kernel):
    kernel.remember("main", "shared baseline fact about servers", kind="fact")
    fork(db, "tester", "main", "child")
    kernel.remember("child", "child-only speculation about servers", kind="fact")

    child_hits = {h.memory.content for h in kernel.recall("child", "servers", k=10)}
    main_hits = {h.memory.content for h in kernel.recall("main", "servers", k=10)}

    assert "child-only speculation about servers" in child_hits
    assert "child-only speculation about servers" not in main_hits
    assert "shared baseline fact about servers" in main_hits


def test_parent_writes_after_fork_are_invisible_to_child(db, kernel):
    kernel.remember("main", "fact known before the fork about servers", kind="fact")
    fork(db, "tester", "main", "child")
    kernel.remember("main", "fact added after the fork about servers", kind="fact")

    child_hits = {h.memory.content for h in kernel.recall("child", "servers", k=10)}
    assert "fact known before the fork about servers" in child_hits
    # The ancestry bound must hide post-fork parent writes.
    assert "fact added after the fork about servers" not in child_hits


def test_supersede_on_branch_leaves_parent_untouched(db, kernel, test_dsn):
    m = kernel.remember("main", "the port is 8080 on the server", kind="fact")
    fork(db, "tester", "main", "child")
    kernel.supersede(m.id, "the port is 9090 on the server", branch="child")

    child_hits = {h.memory.content for h in kernel.recall("child", "port server", k=10)}
    main_hits = {h.memory.content for h in kernel.recall("main", "port server", k=10)}

    assert "the port is 9090 on the server" in child_hits
    assert "the port is 8080 on the server" not in child_hits  # shadowed on child
    assert "the port is 8080 on the server" in main_hits       # parent untouched
    assert "the port is 9090 on the server" not in main_hits

    # The ancestor row itself was never mutated.
    with psycopg.connect(test_dsn) as conn:
        status = conn.execute("SELECT status FROM memories WHERE id = %s", (m.id,)).fetchone()[0]
    assert status == "active"


def test_retract_on_branch_leaves_parent_untouched(db, kernel):
    m = kernel.remember("main", "a questionable fact about servers", kind="fact")
    fork(db, "tester", "main", "child")
    kernel.retract(m.id, reason="wrong on this branch", branch="child")

    child_hits = {h.memory.id for h in kernel.recall("child", "servers", k=10)}
    main_hits = {h.memory.id for h in kernel.recall("main", "servers", k=10)}
    assert m.id not in child_hits
    assert m.id in main_hits


# -- discard ---------------------------------------------------------------

def test_discard_marks_and_never_deletes(db, kernel, test_dsn):
    child = fork(db, "tester", "main", "doomed")
    kernel.remember("doomed", "speculative memory about servers", kind="fact")
    discard(db, "tester", "doomed", reason="bad path")

    assert get_branch(db, "doomed").status == "discarded"
    assert audit_count(test_dsn, op="discard") == 1
    # Nothing hard-deleted.
    with psycopg.connect(test_dsn) as conn:
        n = conn.execute(
            "SELECT count(*) FROM memories WHERE branch_id = %s", (child.id,)
        ).fetchone()[0]
    assert n == 1


def test_cannot_fork_from_discarded_branch(db, kernel):
    fork(db, "tester", "main", "doomed")
    discard(db, "tester", "doomed")
    with pytest.raises(InvalidStateError):
        fork(db, "tester", "doomed", "child-of-doomed")


# -- commit ----------------------------------------------------------------

def test_commit_replays_branch_memories_onto_parent(db, kernel, test_dsn):
    fork(db, "tester", "main", "child")
    kernel.remember("child", "discovered a new fact about servers", kind="fact")

    result = commit(db, "tester", "child")
    assert result.committed is True
    assert len(result.replayed_memory_ids) == 1
    assert get_branch(db, "child").status == "committed"
    assert audit_count(test_dsn, op="commit") == 1

    main_hits = {h.memory.content for h in kernel.recall("main", "servers", k=10)}
    assert "discovered a new fact about servers" in main_hits


def test_commit_replay_creates_new_rows_with_provenance(db, kernel, test_dsn):
    fork(db, "tester", "main", "child")
    m = kernel.remember("child", "a branch fact about servers", kind="fact")
    result = commit(db, "tester", "child")

    with psycopg.connect(test_dsn) as conn:
        origin = conn.execute(
            "SELECT origin_memory_id FROM memories WHERE id = %s",
            (result.replayed_memory_ids[0],),
        ).fetchone()[0]
        # The branch's own row still exists on the branch.
        still_there = conn.execute(
            "SELECT branch_id FROM memories WHERE id = %s", (m.id,)
        ).fetchone()[0]
    assert origin == m.id
    child = get_branch(db, "child")
    assert still_there == child.id


def test_commit_conflict_returns_data_and_is_a_noop(db, kernel, test_dsn):
    m = kernel.remember("main", "the timeout is 30s on the server", kind="fact")
    fork(db, "tester", "main", "child")
    # Branch modifies the inherited memory...
    kernel.supersede(m.id, "the timeout is 60s on the server", branch="child")
    # ...and the parent also changes it after the fork point.
    kernel.supersede(m.id, "the timeout is 45s on the server")

    before_main = len(kernel.list_memories("main"))
    result = commit(db, "tester", "child")

    assert result.committed is False
    assert len(result.conflicts) == 1
    c = result.conflicts[0]
    assert c.memory_id == m.id
    assert c.branch_status == "superseded"
    # No partial state: branch stays open, nothing replayed onto the parent.
    assert result.replayed_memory_ids == []
    assert get_branch(db, "child").status == "open"
    assert len(kernel.list_memories("main")) == before_main


def test_commit_rejects_non_open_branch(db, kernel):
    fork(db, "tester", "main", "child")
    commit(db, "tester", "child")
    with pytest.raises(InvalidStateError):
        commit(db, "tester", "child")


def test_commit_rejects_root_branch(db, kernel):
    with pytest.raises(InvalidStateError):
        commit(db, "tester", "main")


# -- diff ------------------------------------------------------------------

def test_diff_reports_added_superseded_retracted(db, kernel, test_dsn):
    base = kernel.remember("main", "baseline fact about servers", kind="fact")
    fork(db, "tester", "main", "left")
    fork(db, "tester", "main", "right")

    added_left = kernel.remember("left", "left side new fact", kind="fact")
    kernel.supersede(base.id, "left side revision of baseline", branch="left")
    kernel.retract(base.id, reason="right disagrees", branch="right")

    d = diff(db, "tester", "left", "right")
    assert added_left.id in d.a.added
    assert base.id in d.a.superseded
    assert base.id in d.b.retracted
    assert audit_count(test_dsn, op="diff") == 1


# -- the read stays index-accelerated -------------------------------------

def test_ancestry_recall_still_uses_the_vector_index(db, kernel, test_dsn):
    """Regression guard: ancestry resolution must not defeat the vector index.

    Index selection is cost-based, so the table needs enough rows for the index
    to beat a scan. We first confirm a plain single-branch ANN query uses the
    index on this data, then assert the ancestry query does too — comparing
    against that baseline keeps the test from going flaky on planner changes.
    """
    from kernel.embeddings import FakeEmbeddingProvider, vector_literal
    from kernel.recall import build_recall_sql

    emb = FakeEmbeddingProvider()
    qvec = vector_literal(emb.embed("database"))

    # Seed enough rows that the vector index is the cheaper plan. Drop the index
    # for the bulk load and rebuild once — far faster than maintaining it.
    main = get_branch(db, "main")
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP INDEX memories@vec_memories_embedding")
        with conn.cursor() as cur:
            with cur.copy(
                "COPY memories (branch_id, kind, content, embedding) FROM STDIN"
            ) as cp:
                for i in range(3000):
                    t = f"database event number {i}"
                    cp.write_row((main.id, "fact", t, vector_literal(emb.embed(t))))
        conn.execute(
            "CREATE VECTOR INDEX vec_memories_embedding ON memories (branch_id, embedding)"
        )
        conn.execute("ANALYZE memories")

        baseline = "\n".join(
            str(r[0])
            for r in conn.execute(
                f"EXPLAIN SELECT id FROM memories WHERE branch_id = '{main.id}' "
                f"ORDER BY embedding <-> '{qvec}'::VECTOR LIMIT 10"
            ).fetchall()
        )
    if "vector search" not in baseline:
        pytest.skip("planner did not choose the vector index even for a plain ANN query")

    fork(db, "tester", "main", "child")
    segments = resolve_ancestry(db, "child")
    sql = build_recall_sql(segments, qvec, "TRUE")

    # Substitute the parameters as literals (keeping the casts) so EXPLAIN can
    # plan the exact statement the kernel runs.
    literal_sql = sql.replace("%s::UUID", "{BID}::UUID")
    literal_sql = literal_sql.replace("%s::TIMESTAMPTZ", "NULL::TIMESTAMPTZ")
    literal_sql = literal_sql.replace("%s::INT", "0::INT")
    literal_sql = literal_sql.replace("LIMIT %s", "LIMIT 80", 1)
    literal_sql = literal_sql.replace("LIMIT %s", "LIMIT 10")
    ids = [f"'{s.branch_id}'" for s in segments]
    # Fill the branch-id placeholders (inner IN list, then the anc VALUES rows).
    for bid in ids + ids:
        literal_sql = literal_sql.replace("{BID}", bid, 1)

    with psycopg.connect(test_dsn) as conn:
        rows = conn.execute("EXPLAIN " + literal_sql).fetchall()
    plan = "\n".join(str(r[0]) for r in rows)
    assert "vector search" in plan, f"vector index NOT used:\n{plan}"

"""Replay and decision-provenance tests against a live CockroachDB instance."""

from __future__ import annotations

from datetime import timedelta

import psycopg
import pytest

from kernel.branching import fork
from kernel.errors import ReplayWindowExpiredError
from kernel.replay import (
    explain_decision,
    replay_branch_at,
    replay_cluster_at,
    replay_window_bounds,
    rewind_and_rerun,
)
from tests.conftest import audit_count, requires_crdb

pytestmark = requires_crdb


def _now(dsn: str):
    with psycopg.connect(dsn) as conn:
        return conn.execute("SELECT now()").fetchone()[0]


# -- logical replay --------------------------------------------------------

def test_logical_replay_reconstructs_past_state(db, kernel, test_dsn):
    """Replay must show superseded/retracted memories as they were at T."""
    m1 = kernel.remember("main", "the timeout is 30 seconds", kind="fact")
    m2 = kernel.remember("main", "the port is 8080", kind="fact")
    t0 = _now(test_dsn)

    # After t0: supersede one, retract the other.
    m3 = kernel.supersede(m1.id, "the timeout is 60 seconds")
    kernel.retract(m2.id, reason="wrong")
    t1 = _now(test_dsn)

    at_t0 = {m.id for m in replay_branch_at(db, "tester", "main", t0)}
    at_t1 = {m.id for m in replay_branch_at(db, "tester", "main", t1)}

    # At t0 the originals were active and the replacement did not exist.
    assert at_t0 == {m1.id, m2.id}
    # At t1 only the replacement is active.
    assert at_t1 == {m3.id}


def test_logical_replay_respects_branch_ancestry(db, kernel, test_dsn):
    kernel.remember("main", "known before the fork", kind="fact")
    fork(db, "tester", "main", "child")
    kernel.remember("child", "child-only memory", kind="fact")
    kernel.remember("main", "added to main after the fork", kind="fact")
    t = _now(test_dsn)

    contents = {m.content for m in replay_branch_at(db, "tester", "child", t)}
    assert "known before the fork" in contents
    assert "child-only memory" in contents
    assert "added to main after the fork" not in contents


def test_logical_replay_works_for_a_branch_older_than_the_gc_window(db, kernel, test_dsn):
    """Logical replay is not GC-bounded: a 10h-old branch replays fine.

    We backdate rows directly so the scenario is deterministic. Physical replay
    could not answer this at all (see the GC-window test below).
    """
    window = replay_window_bounds(db)
    old = window.latest - timedelta(hours=10)  # well outside the 4h GC window

    with psycopg.connect(test_dsn, autocommit=True) as conn:
        branch_id = conn.execute("SELECT id FROM branches WHERE name='main'").fetchone()[0]
        conn.execute(
            "INSERT INTO memories (branch_id, kind, content, created_at) "
            "VALUES (%s, 'fact', 'ancient memory', %s)",
            (branch_id, old - timedelta(minutes=5)),
        )
        conn.execute(
            "INSERT INTO memories (branch_id, kind, content, created_at) "
            "VALUES (%s, 'fact', 'recent memory', %s)",
            (branch_id, window.latest),
        )

    assert old < window.earliest, "test setup must be outside the physical window"
    contents = {m.content for m in replay_branch_at(db, "tester", "main", old)}
    assert contents == {"ancient memory"}  # 'recent memory' did not exist yet


# -- physical replay + GC honesty -----------------------------------------

def test_replay_window_bounds_reports_the_gc_ttl(db):
    w = replay_window_bounds(db)
    assert w.gc_ttl_seconds > 0
    assert w.earliest < w.latest
    assert w.contains(w.latest)
    assert not w.contains(w.earliest - timedelta(seconds=1))


def test_physical_replay_reads_historical_cluster_state(db, kernel, test_dsn):
    kernel.remember("main", "first memory", kind="fact")
    t = _now(test_dsn)
    kernel.remember("main", "second memory", kind="fact")

    at_t = {m.content for m in replay_cluster_at(db, "tester", t)}
    assert "first memory" in at_t
    assert "second memory" not in at_t  # did not physically exist yet


def test_physical_replay_sees_in_place_edits_logical_replay_cannot(db, kernel, test_dsn):
    """The logical/physical divergence, demonstrated.

    An out-of-band in-place edit leaves no logical trace, so logical replay shows
    the corrected text while physical replay shows the original bytes.
    """
    m = kernel.remember("main", "original text", kind="fact")
    t = _now(test_dsn)
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("UPDATE memories SET content = 'edited text' WHERE id = %s", (m.id,))

    logical = {mm.content for mm in replay_branch_at(db, "tester", "main", t)}
    physical = {mm.content for mm in replay_cluster_at(db, "tester", t)}

    assert logical == {"edited text"}    # the row Recall knows about today
    assert physical == {"original text"}  # what was actually stored at t


def test_physical_replay_past_gc_window_raises_typed_error(db):
    window = replay_window_bounds(db)
    too_old = window.earliest - timedelta(hours=1)
    with pytest.raises(ReplayWindowExpiredError) as exc:
        replay_cluster_at(db, "tester", too_old)
    msg = str(exc.value)
    # The error must name the window rather than failing obscurely.
    assert str(window.gc_ttl_seconds) in msg
    assert "replay_branch_at" in msg  # points at the durable alternative


def test_physical_replay_future_timestamp_raises(db):
    window = replay_window_bounds(db)
    with pytest.raises(ReplayWindowExpiredError):
        replay_cluster_at(db, "tester", window.latest + timedelta(hours=1))


def test_aost_physical_replay_still_uses_the_vector_index(db, kernel, test_dsn):
    """Regression guard: SET TRANSACTION AOST + CTE + ANN keeps the vector index."""
    from kernel.branching import get_branch
    from kernel.embeddings import FakeEmbeddingProvider, vector_literal

    emb = FakeEmbeddingProvider()
    qvec = vector_literal(emb.embed("database"))
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
        ts = conn.execute("SELECT now()").fetchone()[0]

        baseline = "\n".join(
            str(r[0])
            for r in conn.execute(
                f"EXPLAIN SELECT id FROM memories WHERE branch_id = '{main.id}' "
                f"ORDER BY embedding <-> '{qvec}'::VECTOR LIMIT 10"
            ).fetchall()
        )
    if "vector search" not in baseline:
        pytest.skip("planner did not choose the vector index even for a plain ANN query")

    ann = (
        f"WITH candidates AS ("
        f"  SELECT id, embedding <-> '{qvec}'::VECTOR AS d FROM memories "
        f"  WHERE branch_id = '{main.id}' "
        f"  ORDER BY embedding <-> '{qvec}'::VECTOR LIMIT 80) "
        f"SELECT id FROM candidates ORDER BY d LIMIT 10"
    )
    with psycopg.connect(test_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET TRANSACTION AS OF SYSTEM TIME '{ts.isoformat()}'")
            cur.execute("EXPLAIN " + ann)
            plan = "\n".join(str(r[0]) for r in cur.fetchall())
        conn.commit()
    assert "vector search" in plan, f"AOST replay lost the vector index:\n{plan}"


# -- explain_decision ------------------------------------------------------

def test_explain_decision_flags_a_retracted_contributing_memory(db, kernel, test_dsn):
    good = kernel.remember("main", "the database is on host alpha", kind="fact")
    bad = kernel.remember("main", "the database is healthy", kind="fact")
    hits = kernel.recall("main", "database", k=5)
    decision = kernel.record_decision(
        "main", agent_id="triage", action="no-op", rationale="looks fine", recalled=hits
    )

    # We later learn one supporting memory was wrong.
    kernel.retract(bad.id, reason="it was not healthy at all")

    explanation = explain_decision(db, "tester", decision.id)

    assert explanation.decision.id == decision.id
    assert explanation.branch_name == "main"
    assert explanation.has_invalidated_memories is True
    assert explanation.invalidated_count == 1
    assert explanation.invalidated_memory_ids == [bad.id]

    by_id = {c.memory_id: c for c in explanation.memories}
    assert by_id[bad.id].retracted is True
    assert by_id[bad.id].invalidated is True
    assert by_id[bad.id].status_now == "retracted"
    assert by_id[bad.id].retracted_at is not None
    assert by_id[good.id].invalidated is False
    assert by_id[good.id].status_now == "active"
    # Similarity and rank as recorded at decision time.
    assert all(c.similarity is not None for c in explanation.memories)
    assert [c.rank for c in explanation.memories] == sorted(
        c.rank for c in explanation.memories
    )
    assert audit_count(test_dsn, op="explain_decision") == 1


def test_explain_decision_flags_superseded_memory(db, kernel):
    m = kernel.remember("main", "the retry limit is 3", kind="fact")
    hits = kernel.recall("main", "retry limit", k=5)
    decision = kernel.record_decision(
        "main", agent_id="triage", action="retry", recalled=hits
    )
    kernel.supersede(m.id, "the retry limit is 10")

    explanation = explain_decision(db, "tester", decision.id)
    entry = {c.memory_id: c for c in explanation.memories}[m.id]
    assert entry.superseded is True
    assert entry.invalidated is True
    assert entry.superseded_by is not None
    assert explanation.has_invalidated_memories is True


def test_explain_decision_clean_when_nothing_changed(db, kernel):
    kernel.remember("main", "a stable fact", kind="fact")
    hits = kernel.recall("main", "stable", k=5)
    decision = kernel.record_decision("main", agent_id="a", action="act", recalled=hits)
    explanation = explain_decision(db, "tester", decision.id)
    assert explanation.has_invalidated_memories is False
    assert explanation.invalidated_count == 0


# -- rewind_and_rerun ------------------------------------------------------

def test_rewind_and_rerun_produces_a_different_action(db, kernel, test_dsn):
    """A fake agent that reacts to the presence of a memory must flip its action."""
    healthy = kernel.remember("main", "the database is healthy", kind="fact")
    kernel.remember("main", "checkout latency is elevated", kind="metric")
    hits = kernel.recall("main", "database", k=5)
    decision = kernel.record_decision(
        "main",
        agent_id="triage",
        action="no-op",
        rationale="database reported healthy",
        recalled=hits,
    )

    def agent(ctx):
        # Decide purely from the reconstructed context.
        if any("healthy" in m.content for m in ctx.memories):
            return {"action": "no-op", "rationale": "database reported healthy"}
        return {"action": "page-oncall", "rationale": "no evidence the database is healthy"}

    # Re-run against the state as it was: the agent still says no-op.
    same = rewind_and_rerun(db, "tester", decision.id, agent)
    assert same.old_action == "no-op"
    assert same.new_action == "no-op"
    assert same.action_changed is False

    # Now retract the memory that justified it, and re-run again.
    kernel.retract(healthy.id, reason="the database was not healthy")
    changed = rewind_and_rerun(db, "tester", decision.id, agent)

    assert changed.old_action == "no-op"
    assert changed.new_action == "no-op", (
        "replay reconstructs the PAST state, so the past decision must reproduce"
    )

    # Re-running against *today's* state is what flips it — prove the memory
    # really left the current set.
    now_memories = replay_branch_at(db, "tester", "main", _now(test_dsn))
    assert healthy.id not in {m.id for m in now_memories}
    assert changed.memory_diff.then_count > changed.memory_diff.now_count
    assert healthy.id in {r.memory_id for r in changed.memory_diff.only_then}
    assert audit_count(test_dsn, op="rewind_and_rerun") == 2


def test_rewind_and_rerun_against_now_flips_the_action(db, kernel, test_dsn):
    """The money feature: given what we know now, the agent decides differently."""
    healthy = kernel.remember("main", "the database is healthy", kind="fact")
    kernel.remember("main", "checkout latency is elevated", kind="metric")
    hits = kernel.recall("main", "database", k=5)
    decision = kernel.record_decision(
        "main", agent_id="triage", action="no-op", recalled=hits
    )

    def agent(ctx):
        if any("healthy" in m.content for m in ctx.memories):
            return {"action": "no-op", "rationale": "database reported healthy"}
        return {"action": "page-oncall", "rationale": "no evidence the database is healthy"}

    kernel.retract(healthy.id, reason="the database was not healthy")

    # Faithful replay at decision time still reproduces the original action...
    faithful = rewind_and_rerun(db, "tester", decision.id, agent)
    assert faithful.new_action == "no-op"
    assert faithful.action_changed is False
    assert faithful.evaluated_at == faithful.decision_at

    # ...but re-run against today's knowledge, it decides differently.
    today = rewind_and_rerun(
        db, "tester", decision.id, agent, as_of=replay_window_bounds(db).latest
    )
    assert today.new_action == "page-oncall"
    assert today.action_changed is True
    assert today.evaluated_at > today.decision_at
    assert healthy.id in {r.memory_id for r in today.memory_diff.only_then}


def test_rewind_and_rerun_agent_sees_only_past_memories(db, kernel, test_dsn):
    kernel.remember("main", "known at decision time", kind="fact")
    hits = kernel.recall("main", "known", k=5)
    decision = kernel.record_decision("main", agent_id="a", action="old", recalled=hits)
    kernel.remember("main", "learned after the decision", kind="fact")

    seen: list[str] = []

    def agent(ctx):
        seen.extend(m.content for m in ctx.memories)
        return "new" if len(ctx.memories) > 1 else "old"

    result = rewind_and_rerun(db, "tester", decision.id, agent)
    assert "known at decision time" in seen
    assert "learned after the decision" not in seen  # not yet known
    assert result.new_action == "old"
    assert result.action_changed is False
    # The diff surfaces what has been learned since.
    assert "learned after the decision" in {r.content for r in result.memory_diff.only_now}


def test_rewind_and_rerun_accepts_plain_string_and_reports_diff(db, kernel):
    kernel.remember("main", "a fact", kind="fact")
    hits = kernel.recall("main", "fact", k=5)
    decision = kernel.record_decision("main", agent_id="a", action="old", recalled=hits)

    result = rewind_and_rerun(db, "tester", decision.id, lambda ctx: "brand-new-action")
    assert result.new_action == "brand-new-action"
    assert result.action_changed is True
    assert result.contributing_memory_ids
    assert result.memory_diff.then_count >= 1

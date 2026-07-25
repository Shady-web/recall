"""Memory kernel tests against a live CockroachDB instance.

Covers the Phase 1 invariants:
* every write path produces exactly one audit row,
* a failed write produces zero audit rows (and no partial data),
* supersede preserves history,
* read-only mode blocks writes but still allows (audited) reads,
* decision provenance rows are written.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from kernel.errors import InvalidStateError, NotFoundError, ReadOnlyError
from kernel.memory import MemoryKernel
from tests.conftest import audit_count, requires_crdb, table_count

pytestmark = requires_crdb


# -- write paths each produce exactly one audit row -----------------------

def test_remember_writes_exactly_one_audit_row(kernel, test_dsn):
    assert audit_count(test_dsn) == 0
    m = kernel.remember("main", "the sky is blue", kind="fact", source="obs")
    assert audit_count(test_dsn) == 1
    assert audit_count(test_dsn, op="remember") == 1
    # The audit row targets the new memory.
    with psycopg.connect(test_dsn) as conn:
        target = conn.execute(
            "SELECT target_id FROM audit_log WHERE op = 'remember'"
        ).fetchone()[0]
    assert target == m.id


def test_supersede_writes_exactly_one_audit_row(kernel, test_dsn):
    m = kernel.remember("main", "v1", kind="fact")
    before = audit_count(test_dsn)
    kernel.supersede(m.id, "v2")
    assert audit_count(test_dsn) - before == 1
    assert audit_count(test_dsn, op="supersede") == 1


def test_retract_writes_exactly_one_audit_row(kernel, test_dsn):
    m = kernel.remember("main", "temporary", kind="fact")
    before = audit_count(test_dsn)
    kernel.retract(m.id, reason="was wrong")
    assert audit_count(test_dsn) - before == 1
    assert audit_count(test_dsn, op="retract") == 1


def test_record_decision_writes_exactly_one_audit_row(kernel, test_dsn):
    m1 = kernel.remember("main", "cpu at 99%", kind="metric")
    m2 = kernel.remember("main", "deploy 10m ago", kind="event")
    before = audit_count(test_dsn)
    d = kernel.record_decision(
        "main",
        agent_id="triage-1",
        action="rollback",
        rationale="spike right after deploy",
        memory_ids=[m1.id, m2.id],
    )
    assert audit_count(test_dsn) - before == 1
    assert audit_count(test_dsn, op="record_decision") == 1
    # Provenance rows exist with the ranks we assigned.
    with psycopg.connect(test_dsn) as conn:
        rows = conn.execute(
            "SELECT memory_id, rank FROM decision_memories "
            "WHERE decision_id = %s ORDER BY rank",
            (d.id,),
        ).fetchall()
    assert [r[1] for r in rows] == [0, 1]
    assert {r[0] for r in rows} == {m1.id, m2.id}


# -- reads are audited too (per CONTEXT §4) -------------------------------

def test_get_and_list_are_audited(kernel, test_dsn):
    m = kernel.remember("main", "fact one", kind="fact")
    base = audit_count(test_dsn)
    got = kernel.get(m.id)
    assert got.id == m.id
    assert audit_count(test_dsn, op="get") == 1

    memories = kernel.list_memories("main")
    assert len(memories) == 1
    assert audit_count(test_dsn, op="list") == 1
    assert audit_count(test_dsn) == base + 2


# -- a failed write produces zero audit rows and no partial data ----------

def test_remember_on_missing_branch_writes_nothing(kernel, test_dsn):
    with pytest.raises(NotFoundError):
        kernel.remember("does-not-exist", "orphan", kind="fact")
    assert audit_count(test_dsn) == 0
    assert table_count(test_dsn, "memories") == 0


def test_record_decision_bad_memory_rolls_back_entirely(kernel, test_dsn):
    # The decision row inserts, then the provenance FK fails — the whole
    # transaction (decision + its audit row) must roll back.
    before_decisions = table_count(test_dsn, "decisions")
    before_audit = audit_count(test_dsn)
    with pytest.raises(psycopg.Error):
        kernel.record_decision(
            "main",
            agent_id="triage-1",
            action="noop",
            memory_ids=[uuid.uuid4()],  # references a non-existent memory
        )
    assert table_count(test_dsn, "decisions") == before_decisions
    assert audit_count(test_dsn) == before_audit


# -- supersede preserves history -----------------------------------------

def test_supersede_preserves_history(kernel, test_dsn):
    m1 = kernel.remember("main", "the port is 8080", kind="fact")
    m2 = kernel.supersede(m1.id, "the port is 9090")

    old = kernel.get(m1.id)
    new = kernel.get(m2.id)

    # Old row is retained, marked superseded, and linked to the replacement.
    assert old.status == "superseded"
    assert old.superseded_by == m2.id
    assert old.content == "the port is 8080"
    # New row is active with the new content, on the same branch.
    assert new.status == "active"
    assert new.content == "the port is 9090"
    assert new.branch_id == old.branch_id
    # Both rows physically exist — nothing was deleted.
    assert table_count(test_dsn, "memories") == 2


def test_supersede_non_active_is_rejected(kernel):
    m = kernel.remember("main", "v1", kind="fact")
    kernel.retract(m.id, reason="oops")
    with pytest.raises(InvalidStateError):
        kernel.supersede(m.id, "v2")


# -- read-only mode -------------------------------------------------------

def test_read_only_blocks_writes(db, test_dsn):
    ro = MemoryKernel(db, actor="tester", read_only=True)
    with pytest.raises(ReadOnlyError):
        ro.remember("main", "should not persist", kind="fact")
    assert audit_count(test_dsn) == 0
    assert table_count(test_dsn, "memories") == 0


def test_read_only_allows_audited_reads(db, kernel, test_dsn):
    kernel.remember("main", "readable", kind="fact")  # seed via a writable kernel
    ro = MemoryKernel(db, actor="reader", read_only=True)
    memories = ro.list_memories("main")
    assert len(memories) == 1
    # The read itself is audited, attributed to the read-only actor.
    with psycopg.connect(test_dsn) as conn:
        actor = conn.execute(
            "SELECT actor FROM audit_log WHERE op = 'list'"
        ).fetchone()[0]
    assert actor == "reader"


def test_kernel_requires_actor(db):
    with pytest.raises(ValueError):
        MemoryKernel(db, actor="")

"""Recall + embed-on-write tests against a live CockroachDB instance.

Uses the deterministic FakeEmbeddingProvider (via the ``kernel`` fixture) so the
suite runs with no AWS credentials or cost.
"""

from __future__ import annotations

import pytest

from kernel.embeddings import FakeEmbeddingProvider
from kernel.errors import EmbeddingError
from kernel.memory import MemoryKernel
from tests.conftest import audit_count, requires_crdb, table_count

pytestmark = requires_crdb


class _RaisingEmbedder(FakeEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        raise EmbeddingError("simulated embedding failure")


# -- embed-on-write -------------------------------------------------------

def test_remember_stores_an_embedding(kernel, test_dsn):
    import psycopg

    m = kernel.remember("main", "the sky is blue", kind="fact")
    with psycopg.connect(test_dsn) as conn:
        embedding = conn.execute(
            "SELECT embedding FROM memories WHERE id = %s", (m.id,)
        ).fetchone()[0]
    assert embedding is not None  # never stored unembedded


def test_failed_embedding_rolls_back_the_write(db, test_dsn):
    k = MemoryKernel(db, actor="tester", read_only=False, embedder=_RaisingEmbedder())
    with pytest.raises(EmbeddingError):
        k.remember("main", "should never persist", kind="fact")
    # No memory and no audit row: the write did not happen at all.
    assert table_count(test_dsn, "memories") == 0
    assert audit_count(test_dsn) == 0


# -- semantic ranking -----------------------------------------------------

def test_recall_ranks_related_above_unrelated(kernel):
    kernel.remember("main", "the production database server crashed overnight", kind="fact")
    kernel.remember("main", "i had a pasta lunch with my colleague", kind="fact")
    kernel.remember("main", "the CI pipeline is flaky on tuesdays", kind="fact")

    results = kernel.recall("main", "database server crash", k=3)
    assert len(results) >= 1
    # The database memory should be the top hit.
    assert "database server" in results[0].memory.content
    # Ranks are 1-based and ascending; similarities are non-increasing.
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


# -- filters actually filter ----------------------------------------------

def test_recall_kind_filter(kernel):
    kernel.remember("main", "database server crashed", kind="incident")
    kernel.remember("main", "database server nominal", kind="metric")
    results = kernel.recall("main", "database server", k=10, kind="incident")
    assert len(results) == 1
    assert results[0].memory.kind == "incident"


def test_recall_min_confidence_filter(kernel):
    kernel.remember("main", "database maybe down", kind="fact", confidence=0.2)
    kernel.remember("main", "database definitely down", kind="fact", confidence=0.9)
    results = kernel.recall("main", "database down", k=10, min_confidence=0.5)
    assert len(results) == 1
    assert results[0].memory.confidence >= 0.5


def test_recall_status_filter_excludes_retracted_by_default(kernel):
    m = kernel.remember("main", "database on fire", kind="fact")
    kernel.remember("main", "database is fine", kind="fact")
    kernel.retract(m.id, reason="false alarm")
    # Default status='active' should exclude the retracted memory.
    results = kernel.recall("main", "database", k=10)
    ids = {r.memory.id for r in results}
    assert m.id not in ids
    # Explicitly asking for retracted finds it.
    retracted = kernel.recall("main", "database", k=10, status="retracted")
    assert m.id in {r.memory.id for r in retracted}


# -- every recall is audited ---------------------------------------------

def test_recall_writes_one_audit_row(kernel, test_dsn):
    import psycopg

    kernel.remember("main", "database server crashed", kind="fact")
    before = audit_count(test_dsn)
    results = kernel.recall("main", "database", k=5)
    assert audit_count(test_dsn) - before == 1
    assert audit_count(test_dsn, op="recall") == 1
    with psycopg.connect(test_dsn) as conn:
        payload = conn.execute(
            "SELECT payload FROM audit_log WHERE op = 'recall'"
        ).fetchone()[0]
    assert payload["query"] == "database"
    assert payload["returned_ids"] == [str(r.memory.id) for r in results]


def test_recall_allowed_in_read_only_mode(db, kernel, test_dsn):
    kernel.remember("main", "readable memory about servers", kind="fact")
    ro = MemoryKernel(
        db, actor="reader", read_only=True, embedder=FakeEmbeddingProvider()
    )
    results = ro.recall("main", "servers", k=5)
    assert len(results) >= 1

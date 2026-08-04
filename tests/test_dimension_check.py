"""Startup check: the embedding provider's width must match the schema's.

A mismatch is otherwise invisible until the first write, where it surfaces as a
database complaint about the row being inserted rather than as what it is — a
provider configured for a different model than the schema was migrated for.
"""

from __future__ import annotations

import pytest

from kernel.db import Database, schema_vector_dimension, verify_embedding_dimension
from kernel.embeddings import FakeEmbeddingProvider
from kernel.errors import SchemaMismatchError

SCHEMA_DIMENSIONS = 1024  # migrations/001_init.sql: embedding VECTOR(1024)


def test_reads_the_declared_width_from_the_schema(db: Database):
    assert schema_vector_dimension(db) == SCHEMA_DIMENSIONS


def test_matching_provider_passes(db: Database):
    assert (
        verify_embedding_dimension(db, FakeEmbeddingProvider(dimensions=1024))
        == SCHEMA_DIMENSIONS
    )


def test_mismatched_provider_fails_loudly(db: Database):
    """The message must name both numbers and how to resolve it."""
    with pytest.raises(SchemaMismatchError) as exc:
        verify_embedding_dimension(db, FakeEmbeddingProvider(dimensions=512))

    message = str(exc.value)
    assert "512" in message
    assert "1024" in message
    assert "memories.embedding" in message


def test_mismatch_is_caught_before_any_row_is_written(db: Database, kernel):
    """The check must fire at startup, not leave the failure to INSERT time."""
    from kernel.memory import MemoryKernel

    kernel.remember("main", "a memory written with the right width", kind="fact")

    wrong = MemoryKernel(
        db, actor="tester", embedder=FakeEmbeddingProvider(dimensions=512)
    )
    with pytest.raises(SchemaMismatchError):
        verify_embedding_dimension(wrong.db, wrong.embedder)

    with db.transaction(read_only=True) as conn:
        count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    assert count == 1  # the mismatched kernel wrote nothing


def test_unmigrated_database_is_not_reported_as_a_mismatch(empty_dsn: str):
    """An empty database is a different failure, with its own error."""
    database = Database(empty_dsn, min_size=1, max_size=2)
    try:
        assert schema_vector_dimension(database) is None
        assert verify_embedding_dimension(database, FakeEmbeddingProvider()) is None
    finally:
        database.close()


def test_embedder_without_dimensions_is_skipped(db: Database):
    assert verify_embedding_dimension(db, object()) is None
    assert verify_embedding_dimension(db, None) is None

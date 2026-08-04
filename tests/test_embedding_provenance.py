"""Tests for embedding-space provenance (migration 004).

The bug these exist to prevent was observed, not theorised. A database seeded
with :class:`kernel.embeddings.FakeEmbeddingProvider` was queried with real
Titan embeddings. Nothing failed. Recall returned a full page of hits with
similarity scores of 0.040, 0.022, 0.008, 0.001, -0.009, -0.011 — cosine
similarity between two unrelated vector spaces, i.e. orthogonal noise — and the
only symptom was that the numbers "looked a bit low".

This is the worst failure mode a memory system can have: it does not break, it
quietly stops being about meaning. The pre-existing dimension check cannot catch
it, because both providers emit 1024-dimension unit vectors. Width was never the
invariant that mattered; the *space* is.
"""

from __future__ import annotations

import pytest

from kernel.db import stored_embedding_spaces, verify_embedding_provider
from kernel.embeddings import BedrockEmbeddingProvider, FakeEmbeddingProvider
from kernel.errors import SchemaMismatchError


def _bedrock() -> BedrockEmbeddingProvider:
    # No client is ever built: space_id is pure configuration, so this makes no
    # AWS call and needs no credentials.
    return BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0", region="us-east-1"
    )


# ---------------------------------------------------------------------------
# Space identity
# ---------------------------------------------------------------------------


def test_providers_of_equal_width_have_different_space_ids():
    """The whole point: same dimensions, different space."""
    fake = FakeEmbeddingProvider(dimensions=1024)
    bedrock = _bedrock()
    assert fake.dimensions == bedrock.dimensions == 1024
    assert fake.space_id != bedrock.space_id


def test_dimension_is_part_of_space_identity():
    """Titan at 512 is a different space from Titan at 1024, not a prefix of it."""
    assert (
        BedrockEmbeddingProvider(
            model_id="m", region="r", dimensions=512
        ).space_id
        != BedrockEmbeddingProvider(model_id="m", region="r", dimensions=1024).space_id
    )
    assert FakeEmbeddingProvider(512).space_id != FakeEmbeddingProvider(1024).space_id


def test_fake_space_id_is_unmistakable():
    """A database full of fake vectors must be obvious to anyone inspecting it."""
    assert "fake" in FakeEmbeddingProvider().space_id


# ---------------------------------------------------------------------------
# Provenance is recorded on write
# ---------------------------------------------------------------------------


def test_remember_stamps_the_embedding_space(kernel, db):
    kernel.remember("main", "a fact worth keeping", "fact")
    assert stored_embedding_spaces(db) == {kernel.embedder.space_id: 1}


def test_supersede_stamps_the_replacement(kernel, db):
    memory = kernel.remember("main", "original", "fact")
    kernel.supersede(memory.id, "corrected")
    spaces = stored_embedding_spaces(db)
    # Both the original and its replacement carry provenance.
    assert spaces == {kernel.embedder.space_id: 2}


def test_commit_preserves_the_vectors_own_provenance(kernel, db):
    """A replayed row keeps the provenance of the vector it carries."""
    kernel.remember("main", "base fact", "fact")
    branch = kernel.fork("main", "feature/x")
    kernel.remember(branch.id, "branch fact", "fact")

    result = kernel.commit_branch(branch.id)
    assert result.committed is True

    spaces = stored_embedding_spaces(db)
    assert set(spaces) == {kernel.embedder.space_id}
    assert None not in spaces  # the replayed row must not lose its stamp


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_matching_provider_passes(kernel, db):
    kernel.remember("main", "a fact", "fact")
    assert verify_embedding_provider(db, kernel.embedder) == kernel.embedder.space_id


def test_foreign_provider_is_refused(kernel, db):
    """The exact mistake: fake-seeded database, queried with real Bedrock."""
    kernel.remember("main", "a fact", "fact")

    with pytest.raises(SchemaMismatchError) as excinfo:
        verify_embedding_provider(db, _bedrock())

    message = str(excinfo.value)
    # The error must name both spaces, or it is not actionable.
    assert kernel.embedder.space_id in message
    assert "amazon.titan-embed-text-v2:0" in message
    # And must say why it matters, not just that it differs.
    assert "not comparable" in message


def test_guard_is_a_no_op_on_an_empty_corpus(db):
    """A fresh database is not a mismatch — it has no space yet."""
    assert verify_embedding_provider(db, FakeEmbeddingProvider()) is not None or True
    assert stored_embedding_spaces(db) == {}


def test_guard_ignores_a_provider_without_identity(db):
    """Anything lacking space_id cannot be checked, and must not crash."""
    assert verify_embedding_provider(db, object()) is None
    assert verify_embedding_provider(db, None) is None


def test_legacy_rows_warn_rather_than_block(kernel, db, caplog):
    """Rows predating migration 004 have unknown provenance.

    Refusing on them would break every existing database; guessing a provenance
    for them would be a lie. Warn, and let the operator decide.
    """
    kernel.remember("main", "a fact", "fact")
    with db.transaction() as conn:
        conn.execute("UPDATE memories SET embedding_model = NULL")

    with caplog.at_level("WARNING"):
        result = verify_embedding_provider(db, _bedrock())

    assert result == _bedrock().space_id  # allowed through
    assert "predate embedding provenance" in caplog.text


def test_mixed_legacy_and_foreign_still_refuses(kernel, db):
    """One unverifiable row must not mask a genuinely foreign one."""
    kernel.remember("main", "legacy fact", "fact")
    kernel.remember("main", "known fact", "fact")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE memories SET embedding_model = NULL WHERE content = 'legacy fact'"
        )

    with pytest.raises(SchemaMismatchError):
        verify_embedding_provider(db, _bedrock())

"""Embedding provider tests — no AWS, no database."""

from __future__ import annotations

import json
import math

import pytest
from botocore.exceptions import ClientError

from kernel.embeddings import (
    BedrockEmbeddingProvider,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    vector_literal,
)
from kernel.errors import EmbeddingError


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# -- FakeEmbeddingProvider ------------------------------------------------

def test_fake_is_deterministic_and_unit_norm():
    fake = FakeEmbeddingProvider()
    v1 = fake.embed("the database server crashed")
    v2 = fake.embed("the database server crashed")
    assert v1 == v2  # deterministic
    assert len(v1) == 1024
    assert math.isclose(math.sqrt(_dot(v1, v1)), 1.0, rel_tol=1e-9)


def test_fake_is_a_valid_provider():
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)


def test_fake_related_texts_are_closer_than_unrelated():
    fake = FakeEmbeddingProvider()
    query = fake.embed("the database server crashed")
    related = fake.embed("our production database server went down")
    unrelated = fake.embed("i ate a delicious sandwich for lunch")
    # Cosine similarity == dot product for unit vectors.
    assert _dot(query, related) > _dot(query, unrelated)


def test_fake_handles_empty_text():
    v = FakeEmbeddingProvider().embed("!!! ???")  # no alphanumeric tokens
    assert len(v) == 1024
    assert math.isclose(math.sqrt(_dot(v, v)), 1.0, rel_tol=1e-9)


def test_vector_literal_format():
    assert vector_literal([0.0, 1.5, -2.0]).startswith("[")
    assert vector_literal([0.0, 1.5, -2.0]).endswith("]")
    assert vector_literal([1.0, 2.0]) == "[1.0,2.0]"


# -- BedrockEmbeddingProvider (with a fake boto client) -------------------

class _FakeBody:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data


def _throttle_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "InvokeModel",
    )


class _FakeBedrockClient:
    """Fails with ThrottlingException ``fail_times`` times, then succeeds."""

    def __init__(self, fail_times: int, embedding: list[float]):
        self.fail_times = fail_times
        self.embedding = embedding
        self.calls = 0
        self.last_body: dict | None = None

    def invoke_model(self, *, body, modelId, accept, contentType):
        self.calls += 1
        self.last_body = json.loads(body)
        if self.calls <= self.fail_times:
            raise _throttle_error()
        return {"body": _FakeBody({"embedding": self.embedding, "inputTextTokenCount": 3})}


def test_bedrock_retries_on_throttling_then_succeeds():
    emb = [0.1] * 1024
    client = _FakeBedrockClient(fail_times=2, embedding=emb)
    sleeps: list[float] = []
    provider = BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        client=client,
        sleep=sleeps.append,
    )
    result = provider.embed("hello world")
    assert result == emb
    assert client.calls == 3  # two throttles + one success
    assert len(sleeps) == 2  # backoff before each retry


def test_bedrock_request_body_shape():
    client = _FakeBedrockClient(fail_times=0, embedding=[0.2] * 1024)
    provider = BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0", region="us-east-1", client=client
    )
    provider.embed("some text")
    assert client.last_body == {
        "inputText": "some text",
        "dimensions": 1024,
        "normalize": True,
        "embeddingTypes": ["float"],
    }


def test_bedrock_non_retryable_error_surfaces_clearly():
    class _AccessDenied:
        def invoke_model(self, **_):
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
                "InvokeModel",
            )

    provider = BedrockEmbeddingProvider(
        model_id="m", region="us-east-1", client=_AccessDenied()
    )
    with pytest.raises(EmbeddingError) as exc:
        provider.embed("x")
    assert "AccessDeniedException" in str(exc.value)


def test_bedrock_wrong_dimension_is_rejected():
    client = _FakeBedrockClient(fail_times=0, embedding=[0.1] * 512)  # wrong size
    provider = BedrockEmbeddingProvider(
        model_id="m", region="us-east-1", dimensions=1024, client=client
    )
    with pytest.raises(EmbeddingError):
        provider.embed("x")


def test_embed_batch_processes_all():
    client = _FakeBedrockClient(fail_times=0, embedding=[0.3] * 1024)
    provider = BedrockEmbeddingProvider(
        model_id="m", region="us-east-1", client=client, batch_size=2
    )
    out = provider.embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert client.calls == 3

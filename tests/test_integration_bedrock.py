"""Manual integration test against REAL Amazon Bedrock.

Skipped by default. It costs money and needs AWS credentials, so it only runs
when explicitly enabled:

    RECALL_RUN_BEDROCK_INTEGRATION=1 \\
    BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0 \\
    AWS_REGION=us-east-1 \\
    pytest tests/test_integration_bedrock.py -v

Credentials are resolved by boto3 as usual (env vars, shared config, or an
instance/role profile).
"""

from __future__ import annotations

import math
import os

import pytest

from kernel.embeddings import BedrockEmbeddingProvider

_ENABLED = os.environ.get("RECALL_RUN_BEDROCK_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set RECALL_RUN_BEDROCK_INTEGRATION=1 to run the real Bedrock test",
)


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def test_real_titan_embedding_shape_and_semantics():
    provider = BedrockEmbeddingProvider(dimensions=1024, normalize=True)

    vec = provider.embed("the production database server crashed overnight")
    assert len(vec) == 1024
    # normalize=true → unit vector (this is what makes L2 == cosine ordering).
    assert math.isclose(_norm(vec), 1.0, rel_tol=1e-3)

    related = provider.embed("our database went down in production last night")
    unrelated = provider.embed("i enjoyed a quiet walk on the beach")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cos(vec, related) > cos(vec, unrelated)


def test_real_titan_batch():
    provider = BedrockEmbeddingProvider(dimensions=1024, batch_size=2)
    out = provider.embed_batch(["alpha", "beta", "gamma"])
    assert len(out) == 3
    assert all(len(v) == 1024 for v in out)

"""Embedding providers for Recall.

Defines a small :class:`EmbeddingProvider` protocol and two implementations:

* :class:`BedrockEmbeddingProvider` — calls Amazon Bedrock's Titan Text
  Embeddings V2 model via ``bedrock-runtime`` InvokeModel.
* :class:`FakeEmbeddingProvider` — a deterministic, dependency-free provider so
  the whole test suite (and the benchmark) runs without AWS credentials or cost.

Both return **unit-normalized** vectors, which is what lets the kernel rank by
CockroachDB's L2 (`<->`) vector index and still get cosine-equivalent ordering
(see migrations/002_vector_index.sql for the math).

Titan V2 request/response shape (verified against current AWS docs):

    request body:  {"inputText": str, "dimensions": 1024|512|256,
                    "normalize": bool, "embeddingTypes": ["float"]}
    response body: {"embedding": [float, ...], "inputTextTokenCount": int,
                    "embeddingsByType": {"float": [float, ...]}}

Titan V2 embeds ONE text per InvokeModel call — there is no server-side batch
API — so batching here is client-side chunked iteration (the batch size bounds
how much work is grouped per logical call, a seam for future concurrency).
    ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from kernel.errors import EmbeddingError

logger = logging.getLogger("recall.kernel.embeddings")

# Bedrock/botocore error codes that indicate a transient, retryable condition.
_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)


def _export_bedrock_auth() -> None:
    """Make a ``.env``-supplied Bedrock bearer token visible to botocore.

    Called just before the client is built, not at import, so that constructing
    a provider with an injected client (the whole test suite) never needs a
    loadable configuration. A configuration that will not load is not fatal
    here: boto3 still has its own credential chain to fall back on, and letting
    it try produces a better error than a config traceback would.
    """
    try:
        from kernel.config import settings
    except Exception:  # pragma: no cover - depends on ambient configuration
        logger.debug("no loadable Settings; leaving Bedrock auth to boto3")
        return
    if not settings.export_bedrock_auth():
        logger.debug("no Bedrock bearer token configured; boto3 will use SigV4")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into fixed-length unit vectors."""

    @property
    def dimensions(self) -> int: ...

    @property
    def space_id(self) -> str:
        """Identity of the vector SPACE this provider produces.

        Stored on every memory (``memories.embedding_model``) so a later read can
        prove it is comparing like with like. Two providers may agree on
        dimension and still be mutually meaningless — Titan and the fake provider
        both emit 1024-dim unit vectors — so width is not identity. Any change
        that moves the geometry (a different model, or a different output
        dimension of the same model) must produce a different ``space_id``.
        """
        ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class BedrockEmbeddingProvider:
    """Embeds text with Amazon Titan Text Embeddings V2 via Bedrock.

    The boto3 client is created lazily so importing this module (and running the
    fake-provider test suite) never requires AWS credentials. A client may be
    injected for testing.

    Auth is whatever boto3 resolves. Setting ``AWS_BEARER_TOKEN_BEDROCK`` (a
    Bedrock API key) selects bearer auth over SigV4 automatically, because
    ``bedrock`` is a bearer-capable signing name and this client passes no
    in-code credentials; see :func:`_export_bedrock_auth` for the ``.env`` path.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        region: str | None = None,
        dimensions: int = 1024,
        normalize: bool = True,
        batch_size: int = 16,
        max_attempts: int = 5,
        base_backoff: float = 0.5,
        client: object | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if model_id is None or region is None:
            from kernel.config import settings

            model_id = model_id or settings.bedrock_embedding_model
            region = region or settings.aws_region
        self.model_id = model_id
        self.region = region
        self._dimensions = dimensions
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self._client = client
        self._sleep = sleep

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def space_id(self) -> str:
        # Dimension is part of the identity: Titan V2 at 512 dimensions is a
        # different space from Titan V2 at 1024, not a truncation of it.
        return f"bedrock:{self.model_id}:{self._dimensions}"

    def _get_client(self):
        if self._client is None:
            import boto3

            _export_bedrock_auth()
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def embed(self, text: str) -> list[float]:
        return self._invoke(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in chunks of ``batch_size`` (one call per text)."""
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            for text in chunk:
                results.append(self._invoke(text))
        return results

    def _invoke(self, text: str) -> list[float]:
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dimensions,
                "normalize": self.normalize,
                "embeddingTypes": ["float"],
            }
        )
        response = self._invoke_with_retry(body)
        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding")
        if embedding is None:
            # Defensive: fall back to embeddingsByType.float (always present).
            embedding = payload.get("embeddingsByType", {}).get("float")
        if embedding is None:
            raise EmbeddingError(
                f"Bedrock response contained no float embedding: keys={list(payload)}"
            )
        if len(embedding) != self._dimensions:
            raise EmbeddingError(
                f"expected {self._dimensions}-dim embedding, got {len(embedding)}"
            )
        return embedding

    def _invoke_with_retry(self, body: str):
        from botocore.exceptions import (
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
            TokenRetrievalError,
        )

        # Auth never resolved, so no request was signed. Not retryable: waiting
        # will not conjure credentials. These are BotoCoreError subclasses, not
        # ClientError, so without this they escaped the kernel's error type
        # entirely and surfaced as a bare 'Unable to locate credentials'.
        auth_errors = (
            NoCredentialsError,
            PartialCredentialsError,
            TokenRetrievalError,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                return self._get_client().invoke_model(
                    body=body,
                    modelId=self.model_id,
                    accept="application/json",
                    contentType="application/json",
                )
            except auth_errors as exc:
                raise EmbeddingError(
                    f"Bedrock authentication failed ({type(exc).__name__}): {exc}. "
                    f"Set AWS_BEARER_TOKEN_BEDROCK to a Bedrock API key, or "
                    f"provide SigV4 credentials (AWS_ACCESS_KEY_ID / "
                    f"AWS_SECRET_ACCESS_KEY, an aws-configure profile, or an "
                    f"instance role). See DEV_SETUP.md section 4."
                ) from exc
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                message = exc.response.get("Error", {}).get("Message", str(exc))
                if code in _RETRYABLE_CODES and attempt < self.max_attempts:
                    delay = self.base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Bedrock %s on attempt %d/%d; retrying in %.2fs",
                        code,
                        attempt,
                        self.max_attempts,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                # Non-retryable, or attempts exhausted: surface clearly.
                raise EmbeddingError(
                    f"Bedrock embedding failed ({code or type(exc).__name__}): {message}"
                ) from exc


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider:
    """Deterministic embeddings with no external dependencies.

    Uses the hashing trick: each lowercased alphanumeric token increments one
    dimension chosen by a stable hash of the token, and the vector is
    unit-normalized. Texts that share words therefore land close together in the
    vector space, so semantic (lexical-overlap) recall works in tests — while the
    output is fully reproducible across processes (unlike Python's salted
    ``hash``, we use md5).
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def space_id(self) -> str:
        # Deliberately unmistakable. A database carrying these vectors must be
        # obviously non-production when anyone inspects it.
        return f"fake:hashing-trick:{self._dimensions}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self._dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Empty/tokenless text: return a fixed unit vector rather than zeros.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def vector_literal(vec: list[float]) -> str:
    """Format a vector as a CockroachDB VECTOR literal, e.g. ``[0.1,0.2,...]``."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"

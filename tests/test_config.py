"""Tests for ``kernel.config.Settings``.

These build a ``Settings`` instance directly from an explicit environment dict so
they do not depend on a ``.env`` file being present.
"""

from __future__ import annotations

import os

import pytest

from kernel.config import BEDROCK_BEARER_TOKEN_ENV, Settings


@pytest.fixture(autouse=True)
def _isolate_bearer_token_env():
    """Snapshot and restore the bearer-token variable around every test.

    Two reasons this is autouse rather than per-test: ``export_bedrock_auth``
    writes to ``os.environ`` directly, so monkeypatch has nothing to roll back;
    and ``Settings`` reads ``os.environ`` too, so a developer who exports a real
    token in their shell would otherwise change what these tests observe.
    """
    sentinel = object()
    original = os.environ.get(BEDROCK_BEARER_TOKEN_ENV, sentinel)
    os.environ.pop(BEDROCK_BEARER_TOKEN_ENV, None)
    try:
        yield
    finally:
        if original is sentinel:
            os.environ.pop(BEDROCK_BEARER_TOKEN_ENV, None)
        else:
            os.environ[BEDROCK_BEARER_TOKEN_ENV] = original

_BASE_ENV = {
    "CRDB_CONNECTION_STRING": (
        "postgresql://recall_app:supersecret@example.aws-us-east-1."
        "cockroachlabs.cloud:26257/recall?sslmode=verify-full"
    ),
    "AWS_REGION": "us-west-2",
    "BEDROCK_EMBEDDING_MODEL": "amazon.titan-embed-text-v2:0",
    "BEDROCK_REASONING_MODEL": "anthropic.claude-sonnet-4-5-v1:0",
    "RECALL_ACTOR_ID": "test-actor",
    "RECALL_READ_ONLY": "true",
}


def _settings(**overrides) -> Settings:
    env = {**_BASE_ENV, **overrides}
    # Passing values as kwargs (via aliases) bypasses any ambient .env file.
    return Settings(_env_file=None, **env)


def test_reads_all_fields():
    s = _settings()
    assert s.aws_region == "us-west-2"
    assert s.recall_actor_id == "test-actor"
    assert s.bedrock_embedding_model == "amazon.titan-embed-text-v2:0"


def test_read_only_parses_as_bool():
    assert _settings(RECALL_READ_ONLY="true").recall_read_only is True
    assert _settings(RECALL_READ_ONLY="false").recall_read_only is False
    assert _settings(RECALL_READ_ONLY="0").recall_read_only is False


def test_exports_bearer_token_for_botocore(monkeypatch):
    """The token lives in .env, but botocore only reads os.environ."""
    monkeypatch.delenv(BEDROCK_BEARER_TOKEN_ENV, raising=False)
    s = _settings(AWS_BEARER_TOKEN_BEDROCK="tok-from-dotenv")

    assert s.export_bedrock_auth() is True
    assert os.environ[BEDROCK_BEARER_TOKEN_ENV] == "tok-from-dotenv"


def test_export_leaves_an_existing_env_token_alone(monkeypatch):
    """The shell overrides .env, not the other way round."""
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "tok-from-shell")
    s = _settings(AWS_BEARER_TOKEN_BEDROCK="tok-from-dotenv")

    assert s.export_bedrock_auth() is True
    assert os.environ[BEDROCK_BEARER_TOKEN_ENV] == "tok-from-shell"


def test_export_without_a_token_reports_false(monkeypatch):
    """No token configured: boto3 keeps its SigV4 chain, and we say so."""
    monkeypatch.delenv(BEDROCK_BEARER_TOKEN_ENV, raising=False)

    assert _settings().export_bedrock_auth() is False
    assert BEDROCK_BEARER_TOKEN_ENV not in os.environ


def test_str_never_reveals_the_bearer_token():
    rendered = str(_settings(AWS_BEARER_TOKEN_BEDROCK="tok-from-dotenv"))
    assert "tok-from-dotenv" not in rendered
    assert "bearer-token" in rendered
    assert "aws-credential-chain" in str(_settings())


def test_str_redacts_password():
    rendered = str(_settings())
    assert "supersecret" not in rendered
    assert "recall_app" in rendered  # user is fine to show; password is not
    assert "***" in rendered

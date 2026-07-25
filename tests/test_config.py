"""Tests for ``kernel.config.Settings``.

These build a ``Settings`` instance directly from an explicit environment dict so
they do not depend on a ``.env`` file being present.
"""

from __future__ import annotations

from kernel.config import Settings

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


def test_str_redacts_password():
    rendered = str(_settings())
    assert "supersecret" not in rendered
    assert "recall_app" in rendered  # user is fine to show; password is not
    assert "***" in rendered

"""Runtime configuration for the Recall kernel.

Values are read from the process environment (and, for local development, from a
``.env`` file via python-dotenv / pydantic-settings). Never hard-code secrets
here — the only checked-in configuration file is ``.env.example`` with
placeholder values.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over Recall's environment configuration.

    Instantiating :class:`Settings` reads from the environment. Missing values
    that have no default raise a validation error at startup, which is
    intentional — we want to fail loudly rather than connect to the wrong place.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- CockroachDB ------------------------------------------------------
    crdb_connection_string: str = Field(
        ...,
        alias="CRDB_CONNECTION_STRING",
        description="Full psycopg-compatible connection string for the cluster.",
    )

    # --- AWS / Bedrock ----------------------------------------------------
    aws_region: str = Field(
        "us-east-1",
        alias="AWS_REGION",
        description="AWS region used for Bedrock and S3 calls.",
    )
    bedrock_embedding_model: str = Field(
        "amazon.titan-embed-text-v2:0",
        alias="BEDROCK_EMBEDDING_MODEL",
        description="Bedrock model id used to embed memory content.",
    )
    bedrock_reasoning_model: str = Field(
        "anthropic.claude-sonnet-4-5-v1:0",
        alias="BEDROCK_REASONING_MODEL",
        description="Bedrock model id used for agent reasoning.",
    )

    # --- Recall runtime ---------------------------------------------------
    recall_actor_id: str = Field(
        "local-dev",
        alias="RECALL_ACTOR_ID",
        description="Identity stamped onto every audit_log row this process writes.",
    )
    recall_read_only: bool = Field(
        False,
        alias="RECALL_READ_ONLY",
        description="When true, the kernel refuses all writes and permits only reads.",
    )

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        """Human-readable summary that never reveals the connection secret."""
        return (
            "Settings("
            f"crdb={self._redacted_dsn()}, "
            f"aws_region={self.aws_region!r}, "
            f"embedding_model={self.bedrock_embedding_model!r}, "
            f"reasoning_model={self.bedrock_reasoning_model!r}, "
            f"actor_id={self.recall_actor_id!r}, "
            f"read_only={self.recall_read_only}"
            ")"
        )

    def _redacted_dsn(self) -> str:
        """Return the connection string with any password component masked."""
        dsn = self.crdb_connection_string
        if "@" not in dsn or "://" not in dsn:
            return "***"
        scheme, rest = dsn.split("://", 1)
        creds, _, host = rest.partition("@")
        if ":" in creds:
            user = creds.split(":", 1)[0]
            creds = f"{user}:***"
        return f"{scheme}://{creds}@{host}"


# Module-level singleton. Import as ``from kernel.config import settings``.
settings = Settings()

"""Runtime configuration for the Recall kernel.

Values are read from the process environment (and, for local development, from a
``.env`` file via python-dotenv / pydantic-settings). Never hard-code secrets
here — the only checked-in configuration file is ``.env.example`` with
placeholder values.
"""

from __future__ import annotations

import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The environment variable botocore reads for Bedrock bearer-token auth. The
# name is derived from the service's signing name ('bedrock'), not the client
# name ('bedrock-runtime') — see botocore.utils._get_bearer_env_var_name.
BEDROCK_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"


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
    aws_bearer_token_bedrock: SecretStr | None = Field(
        None,
        alias=BEDROCK_BEARER_TOKEN_ENV,
        description=(
            "Bedrock API key (bearer token). Optional: when unset, boto3 falls "
            "back to its normal SigV4 credential chain."
        ),
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

    def export_bedrock_auth(self) -> bool:
        """Publish the Bedrock bearer token to ``os.environ`` for botocore.

        botocore resolves the token from the *process environment* — the
        ``ScopedEnvTokenProvider`` in ``botocore.tokens`` reads
        ``AWS_BEARER_TOKEN_BEDROCK`` at signing time — but pydantic-settings
        reads ``.env`` into this object only, never into ``os.environ``. Without
        this bridge a token that lives in ``.env`` is invisible to boto3, and
        the client falls through to the SigV4 chain and fails to authenticate.

        A token already exported in the real environment always wins, so the
        shell stays the override. Returns whether botocore will now find one.
        """
        if os.environ.get(BEDROCK_BEARER_TOKEN_ENV):
            return True
        if self.aws_bearer_token_bedrock is None:
            return False
        os.environ[BEDROCK_BEARER_TOKEN_ENV] = (
            self.aws_bearer_token_bedrock.get_secret_value()
        )
        return True

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        """Human-readable summary that never reveals the connection secret."""
        return (
            "Settings("
            f"crdb={self._redacted_dsn()}, "
            f"aws_region={self.aws_region!r}, "
            f"embedding_model={self.bedrock_embedding_model!r}, "
            f"reasoning_model={self.bedrock_reasoning_model!r}, "
            f"bedrock_auth={self._bedrock_auth_style()!r}, "
            f"actor_id={self.recall_actor_id!r}, "
            f"read_only={self.recall_read_only}"
            ")"
        )

    def _bedrock_auth_style(self) -> str:
        """Name the auth style without revealing the token itself."""
        if self.aws_bearer_token_bedrock is not None:
            return "bearer-token"
        return "aws-credential-chain"

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

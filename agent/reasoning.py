"""Reasoning backends for the incident-triage agent.

The agent asks a :class:`Reasoner` one question — *given this incident and these
recalled memories, what should we do?* — and gets back a
:class:`kernel.models.AgentDecision`. Two implementations exist, and which one
ran is always reported back to the caller (and onto the UI) rather than being
hidden:

* :class:`BedrockReasoner` — Amazon Bedrock, using the ``bedrock-runtime``
  **Converse** API with the Claude model id from ``BEDROCK_REASONING_MODEL``.
  This is the real path, used for recorded runs.

* :class:`OfflineRuleReasoner` — a deterministic, dependency-free stand-in that
  reaches its conclusion by reading the *same* memories the model would see.
  It exists for the same reason
  :class:`kernel.embeddings.FakeEmbeddingProvider` does: the whole demo can be
  iterated against a local cluster at zero cost. It is **not** a mock that
  returns a canned answer — it genuinely changes its decision when the
  supporting memories change, which is what makes offline rewind runs
  meaningful.

Why boto3 rather than the Anthropic Bedrock SDK client: this project already
authenticates to Bedrock through botocore (``AWS_BEARER_TOKEN_BEDROCK``, see
:func:`kernel.config.Settings.export_bedrock_auth`), which the Anthropic client
does not read — it resolves SigV4 credentials itself. Reusing the working auth
path keeps one credential story for embeddings and reasoning alike.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from kernel.models import AgentDecision

logger = logging.getLogger("recall.agent.reasoning")


class ReasoningError(RuntimeError):
    """Raised when a reasoning backend cannot produce a decision."""


@runtime_checkable
class Reasoner(Protocol):
    """Anything that can turn an incident plus context into a decision."""

    @property
    def name(self) -> str:
        """Stable identifier reported in the UI, e.g. ``bedrock:<model-id>``."""
        ...

    def decide(self, incident: str, memories: list[Any]) -> AgentDecision: ...


# --------------------------------------------------------------------------
# Shared prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an on-call SRE triaging a live production incident.

You are given the incident text and the memories your organisation has recorded
from previous incidents, runbooks, and current configuration. Decide the single
next action to take.

Rules:
- Base your decision ONLY on the incident and the supplied memories. Do not
  invent facts about the system.
- If a memory contradicts or withdraws another, the more recent one wins.
- Prefer the action that resolves the incident without causing a second one.

Reply with ONLY a JSON object, no prose and no code fence:
{"action": "<one imperative sentence, max 140 chars>",
 "rationale": "<2-3 sentences naming the memories that drove this>"}\
"""


def format_context(memories: list[Any]) -> str:
    """Render recalled memories as the numbered context block the model reads.

    Accepts either :class:`kernel.models.Memory` rows or
    :class:`kernel.models.RecallResult` hits (which carry a similarity score);
    both shapes appear because a first run has scores and a replayed run does
    not.
    """
    lines: list[str] = []
    for i, item in enumerate(memories, start=1):
        memory = getattr(item, "memory", item)
        similarity = getattr(item, "similarity", None)
        score = f" similarity={similarity:.3f}" if similarity is not None else ""
        source = memory.source or "unknown"
        lines.append(
            f"[{i}] ({memory.kind}, source={source}, "
            f"confidence={memory.confidence:.2f}{score})\n{memory.content}"
        )
    return "\n\n".join(lines) if lines else "(no memories available)"


def build_user_prompt(incident: str, memories: list[Any]) -> str:
    return (
        f"INCIDENT\n{incident}\n\n"
        f"MEMORIES AVAILABLE TO YOU\n{format_context(memories)}\n\n"
        f"What is the single next action?"
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_decision(text: str) -> AgentDecision:
    """Parse the model's reply into an :class:`AgentDecision`.

    Tolerates a code fence or stray prose around the object, because a decision
    that is merely wrapped is not a failed run. A reply with no JSON at all is
    an error rather than a silently invented action.
    """
    match = _JSON_RE.search(text)
    if match is None:
        raise ReasoningError(
            f"reasoning model returned no JSON object; got: {text[:300]!r}"
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ReasoningError(
            f"reasoning model returned malformed JSON ({exc}): {match.group(0)[:300]!r}"
        ) from exc
    action = payload.get("action")
    if not action:
        raise ReasoningError(f"reasoning model returned no 'action' field: {payload!r}")
    return AgentDecision(action=str(action).strip(), rationale=payload.get("rationale"))


# --------------------------------------------------------------------------
# Bedrock
# --------------------------------------------------------------------------


def _diagnose(message: str, model_id: str) -> str:
    """Append the *actual* remedy for the Bedrock failures we have hit.

    A generic "check your model id" hint is worse than none when the model id is
    fine. Both cases below were observed against a real account, and they need
    opposite responses — one is a config typo, the other cannot be fixed by
    editing config at all.
    """
    lowered = message.lower()

    if "unsupported countries" in lowered or "supported-countries" in lowered:
        return (
            "\n\nThis is NOT a model-id or permissions problem. Anthropic models on "
            "Bedrock are geo-restricted by the *caller's* location, and this "
            "location is outside the supported set — the same credentials can "
            "still invoke non-Anthropic models and the Bedrock control plane. "
            "Editing BEDROCK_REASONING_MODEL will not help; no Claude id will "
            "work from here. Run from a supported location (or via a network "
            "path that egresses from one), or run the agent with the offline "
            "reasoner (--offline / RECALL_OFFLINE=1). "
            "See https://www.anthropic.com/supported-countries"
        )

    if "model identifier is invalid" in lowered:
        hint = (
            "\n\nEvery current Claude model on Bedrock is INFERENCE_PROFILE-only, "
            "so a bare model id is rejected. Use an inference-profile id with a "
            "routing prefix, e.g. 'us.anthropic.claude-sonnet-5' rather than "
            "'anthropic.claude-sonnet-5'. List what this account can use with:\n"
            "  aws bedrock list-inference-profiles --region <region>"
        )
        if not model_id.startswith(("us.", "eu.", "apac.", "global.")):
            hint += (
                f"\n\nThe configured id {model_id!r} has no routing prefix, which "
                f"is the most likely cause."
            )
        return hint

    if "accessdenied" in lowered or "don't have access" in lowered:
        return (
            "\n\nThe model id looks resolvable but this principal is not entitled "
            "to it. Request model access in the Bedrock console (Model access), "
            "and confirm the key or role allows bedrock:InvokeModel for it."
        )

    return (
        "\n\nCheck that BEDROCK_REASONING_MODEL names an inference profile this "
        "account can invoke in this region (aws bedrock list-inference-profiles)."
    )


class BedrockReasoner:
    """Claude on Amazon Bedrock via the ``bedrock-runtime`` Converse API.

    The client is built lazily so importing this module never requires AWS
    credentials, and a client may be injected for tests.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int = 1024,
        client: object | None = None,
    ) -> None:
        if model_id is None or region is None:
            from kernel.config import settings

            model_id = model_id or settings.bedrock_reasoning_model
            region = region or settings.aws_region
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self._client = client

    @property
    def name(self) -> str:
        return f"bedrock:{self.model_id}"

    def _get_client(self):
        if self._client is None:
            import boto3

            from kernel.embeddings import _export_bedrock_auth

            _export_bedrock_auth()
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def decide(self, incident: str, memories: list[Any]) -> AgentDecision:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = self._get_client().converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": build_user_prompt(incident, memories)}],
                    }
                ],
                inferenceConfig={"maxTokens": self.max_tokens},
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            message = exc.response.get("Error", {}).get("Message", str(exc))
            raise ReasoningError(
                f"Bedrock Converse failed ({code or type(exc).__name__}) for model "
                f"{self.model_id!r} in {self.region!r}: {message}"
                f"{_diagnose(message, self.model_id)}"
            ) from exc
        except BotoCoreError as exc:
            raise ReasoningError(
                f"Bedrock Converse could not be called ({type(exc).__name__}): {exc}. "
                f"Set AWS_BEARER_TOKEN_BEDROCK or provide SigV4 credentials; "
                f"see DEV_SETUP.md section 4."
            ) from exc

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in blocks)
        return parse_decision(text)


# --------------------------------------------------------------------------
# Offline
# --------------------------------------------------------------------------

# Signals the offline reasoner looks for. Each is (label, compiled pattern);
# matching is over the memory content the agent was actually given, so the
# decision moves when the memories move.
_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    ("forbid_restart", re.compile(r"never restart|do not restart|must not be restarted", re.I)),
    ("reload_fix", re.compile(r"sighup|config reload|reload the pooler", re.I)),
    ("restart_fix", re.compile(r"restart the pgbouncer|restart pgbouncer", re.I)),
    ("pool_exhaustion", re.compile(r"max_client_conn|pool exhaustion|connection pool", re.I)),
    ("memory_limit", re.compile(r"oomkill|memory limit|out of memory", re.I)),
    ("cache_eviction", re.compile(r"maxmemory|eviction|allkeys-lru", re.I)),
]


class OfflineRuleReasoner:
    """Deterministic stand-in that decides from the supplied memories.

    Used to iterate the demo against the local cluster at no cost. It reads the
    same context a model would and picks the action the evidence supports, so a
    retraction genuinely flips its answer. Always identifies itself as
    ``offline-rules`` so nothing on screen implies a model call that did not
    happen.
    """

    @property
    def name(self) -> str:
        return "offline-rules"

    def decide(self, incident: str, memories: list[Any]) -> AgentDecision:
        found: dict[str, str] = {}
        for item in memories:
            memory = getattr(item, "memory", item)
            for label, pattern in _SIGNALS:
                if label not in found and pattern.search(memory.content):
                    found[label] = memory.content

        cited = ", ".join(sorted(found)) or "no matching signals"

        if "forbid_restart" in found or "reload_fix" in found:
            return AgentDecision(
                action=(
                    "Raise pgbouncer max_client_conn and apply it with a SIGHUP config "
                    "reload; do not restart the pooler."
                ),
                rationale=(
                    "Recalled memory states the pooler must not be restarted during a "
                    "live incident and that a SIGHUP config reload applies the change "
                    "with zero downtime. Signals matched: "
                    f"{cited}. Decided by the offline rule reasoner."
                ),
            )
        if "restart_fix" in found:
            return AgentDecision(
                action="Restart the pgbouncer pods to clear stuck client connections.",
                rationale=(
                    "The runbook recalled for this symptom prescribes restarting the "
                    "pgbouncer pods to clear stuck client connections and restore "
                    f"latency. Signals matched: {cited}. Decided by the offline rule "
                    "reasoner."
                ),
            )
        if "pool_exhaustion" in found:
            return AgentDecision(
                action=(
                    "Compare pgbouncer max_client_conn against replicas x pool_size "
                    "before scaling further."
                ),
                rationale=(
                    "Recalled memories point at connection-pool exhaustion but carry no "
                    f"remediation runbook. Signals matched: {cited}. Decided by the "
                    "offline rule reasoner."
                ),
            )
        if "memory_limit" in found:
            return AgentDecision(
                action="Raise the container memory limit and roll the affected deployment.",
                rationale=(
                    "Recalled memories describe an OOMKill under load. Signals matched: "
                    f"{cited}. Decided by the offline rule reasoner."
                ),
            )
        if "cache_eviction" in found:
            return AgentDecision(
                action="Raise the Redis maxmemory ceiling and confirm the eviction policy.",
                rationale=(
                    "Recalled memories describe cache evictions cascading into errors. "
                    f"Signals matched: {cited}. Decided by the offline rule reasoner."
                ),
            )
        return AgentDecision(
            action="Escalate to the service owner; no recalled memory covers this symptom.",
            rationale=(
                "No recalled memory matched a known remediation for this incident. "
                "Decided by the offline rule reasoner."
            ),
        )


def build_reasoner(offline: bool = False) -> Reasoner:
    """Return the reasoner selected by ``offline``.

    Kept explicit rather than sniffing the environment: which brain produced a
    decision is reported on screen, so it should be a deliberate choice at the
    call site.
    """
    return OfflineRuleReasoner() if offline else BedrockReasoner()

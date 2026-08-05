"""The Recall demo agent: DevOps incident triage.

One run does exactly what the pitch claims, in this order:

1. **Recall** the memories most relevant to the incident, from ``main``.
2. **Fork** a branch so the reasoning that follows cannot pollute ``main``.
3. **Record** the incident itself as a memory on the fork.
4. **Reason** with Bedrock (or the offline rule reasoner) over the recalled set.
5. **Record the decision** with the exact memories used — including each one's
   similarity and rank *at decision time* — as provenance.
6. Optionally **commit** the branch back into ``main``.

Everything the agent knows, it learns through :class:`kernel.memory.MemoryKernel`.
There is no SQL in this module and there never should be (CONTEXT.md §9: the
kernel is never bypassed). Phase 7 wraps this module in a Lambda handler; it is
deliberately a plain callable so that wrapper stays thin.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent.reasoning import Reasoner
from kernel.memory import MemoryKernel
from kernel.models import AgentDecision, Branch, Decision, Memory, RecallResult

logger = logging.getLogger("recall.agent.triage")

#: Identity recorded on every decision this agent writes.
AGENT_ID = "incident-triage"

#: Metadata marker on the memory holding the incident text. The rewind path
#: reads the incident back out of replayed state via this marker rather than
#: being handed it out of band, so a re-run sees exactly what the branch held.
TRIGGER_ROLE = "incident-trigger"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 32) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "incident"


def branch_name_for(incident: str, at: datetime | None = None) -> str:
    """Readable, collision-free branch name: ``incident/<slug>-<stamp>-<nonce>``.

    The timestamp is for humans reading the branch tree; the nonce is what
    actually guarantees uniqueness. A second-resolution stamp alone is not
    enough: ``branches.name`` carries a unique index, and two triage runs of the
    same incident inside one second — a double-clicked button, a retried
    request, two agents reacting to one page — collide and the second one fails
    with a raw ``UniqueViolation``. Observed, not hypothetical.
    """
    at = at or datetime.now(UTC)
    nonce = uuid.uuid4().hex[:6]
    return f"incident/{_slug(incident)}-{at.strftime('%Y%m%dT%H%M%S')}-{nonce}"


@dataclass
class TriageResult:
    """Everything one triage run produced, for the API and the CLI to report."""

    incident: str
    branch: Branch
    decision: Decision
    action: str
    rationale: str | None
    recalled: list[RecallResult] = field(default_factory=list)
    incident_memory: Memory | None = None
    reasoner: str = ""
    committed: bool = False
    commit_conflicts: int = 0

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary. Used by the HTTP layer and the CLI alike."""
        return {
            "incident": self.incident,
            "branch": self.branch.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json"),
            "action": self.action,
            "rationale": self.rationale,
            "reasoner": self.reasoner,
            "committed": self.committed,
            "commit_conflicts": self.commit_conflicts,
            "recalled": [r.model_dump(mode="json") for r in self.recalled],
            "incident_memory": (
                self.incident_memory.model_dump(mode="json")
                if self.incident_memory is not None
                else None
            ),
        }


class IncidentTriageAgent:
    """Triage incidents against branchable memory.

    ``kernel`` and ``reasoner`` are both injected: the kernel because it owns
    the database and the audit identity, the reasoner because whether this run
    cost money is the caller's decision, not a hidden default.
    """

    def __init__(
        self,
        kernel: MemoryKernel,
        reasoner: Reasoner,
        *,
        base_branch: str = "main",
        k: int = 6,
    ) -> None:
        self.kernel = kernel
        self.reasoner = reasoner
        self.base_branch = base_branch
        self.k = k

    # -- the run ----------------------------------------------------------

    def run(
        self,
        incident: str,
        *,
        commit: bool = False,
        branch_name: str | None = None,
    ) -> TriageResult:
        """Triage ``incident`` end to end. See the module docstring for order."""
        logger.info("triage starting (reasoner=%s, base=%s)", self.reasoner.name, self.base_branch)

        # 1. What do we already know? Recalled from the base branch, before the
        #    fork exists, so the fork inherits exactly this state.
        recalled = self.kernel.recall(self.base_branch, incident, k=self.k)
        logger.info("recalled %d memories from %s", len(recalled), self.base_branch)

        # 2. Fork, so everything below is speculative and reversible.
        name = branch_name or branch_name_for(incident)
        branch = self.kernel.fork(self.base_branch, name)
        logger.info("forked %s from %s", branch.name, self.base_branch)

        # 3. The incident is itself a fact worth remembering, and it lands on
        #    the fork rather than on main.
        incident_memory = self.kernel.remember(
            branch.id,
            incident,
            kind="incident",
            source="pagerduty",
            confidence=1.0,
            metadata={"role": TRIGGER_ROLE, "agent_id": AGENT_ID},
        )

        # 4. Reason over the recalled set.
        #
        #    A reasoning failure here (a Bedrock outage, a geo-restricted model,
        #    a malformed reply) leaves a fork that was opened for work that never
        #    happened. Left alone it lingers in the branch tree as an open branch
        #    with an incident and no decision, which reads as "triage in
        #    progress" forever. Discarding it says what actually happened —
        #    speculation abandoned — and discard is a status change, so the
        #    branch and its incident memory stay fully readable for audit.
        #    Observed against a live account, not hypothetical.
        try:
            decision = self.reasoner.decide(incident, recalled)
        except Exception as exc:
            logger.warning(
                "reasoning failed on %s (%s); discarding the speculative branch",
                branch.name,
                type(exc).__name__,
            )
            try:
                self.kernel.discard_branch(
                    branch.id, f"reasoning failed: {type(exc).__name__}: {exc}"
                )
            except Exception:
                # Never let cleanup mask the real failure.
                logger.exception("could not discard %s after a reasoning failure", branch.name)
            raise

        # 5. Record the decision with provenance. Passing `recalled` (not bare
        #    ids) is what captures similarity and rank as of this moment, which
        #    is what explain_decision reports back later.
        recorded = self.kernel.record_decision(
            branch.id,
            agent_id=AGENT_ID,
            action=decision.action,
            rationale=decision.rationale,
            recalled=recalled,
        )
        logger.info("recorded decision %s: %s", recorded.id, decision.action)

        result = TriageResult(
            incident=incident,
            branch=branch,
            decision=recorded,
            action=decision.action,
            rationale=decision.rationale,
            recalled=recalled,
            incident_memory=incident_memory,
            reasoner=self.reasoner.name,
        )

        # 6. Optionally fold the branch back into main. A conflicting commit is
        #    an outcome, not an exception — it is reported, not raised.
        if commit:
            outcome = self.kernel.commit_branch(branch.id)
            result.committed = bool(outcome.committed)
            result.commit_conflicts = len(outcome.conflicts)
            logger.info(
                "commit of %s: committed=%s conflicts=%d",
                branch.name,
                result.committed,
                result.commit_conflicts,
            )

        return result

    # -- rewind -----------------------------------------------------------

    def rerun_callable(self):
        """An agent callable for :func:`kernel.replay.rewind_and_rerun`.

        The kernel hands it a :class:`kernel.models.RerunContext` holding the
        branch state reconstructed at some timestamp. The incident text is read
        back out of that reconstructed state (the memory tagged
        :data:`TRIGGER_ROLE`), so the re-run is driven entirely by what the
        branch contained — never by anything smuggled in from the present.
        """

        def run_against(ctx) -> AgentDecision:
            incident = ""
            context: list[Memory] = []
            for memory in ctx.memories:
                if memory.metadata.get("role") == TRIGGER_ROLE and not incident:
                    incident = memory.content
                else:
                    context.append(memory)
            if not incident:
                # The trigger memory is outside the replayed window (e.g. a
                # timestamp before it was written). Say so rather than
                # reasoning about an empty incident.
                incident = ctx.decision.action
                logger.warning(
                    "no incident-trigger memory in replayed state for decision %s; "
                    "falling back to the recorded action as the prompt",
                    ctx.decision.id,
                )
            return self.reasoner.decide(incident, context)

        return run_against


def triage(
    kernel: MemoryKernel,
    reasoner: Reasoner,
    incident: str,
    *,
    base_branch: str = "main",
    commit: bool = False,
    k: int = 6,
) -> TriageResult:
    """One-shot convenience wrapper over :class:`IncidentTriageAgent`."""
    return IncidentTriageAgent(kernel, reasoner, base_branch=base_branch, k=k).run(
        incident, commit=commit
    )


def retract_memory(
    kernel: MemoryKernel,
    memory_id: str | uuid.UUID,
    reason: str,
    *,
    branch: str | uuid.UUID | None = None,
) -> Memory:
    """Withdraw a memory that turned out to be wrong.

    Thin on purpose — it is the kernel's ``retract`` with the agent's framing.
    It lives here because "we have learned this was wrong" is an operational
    act in the demo narrative, not a database detail.
    """
    return kernel.retract(memory_id, reason, branch=branch)

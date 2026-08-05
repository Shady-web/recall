"""Recall demo agent — DevOps incident triage.

A plain Python module (Phase 7 wraps it in Lambda). It reaches memory only
through :class:`kernel.memory.MemoryKernel`, never CockroachDB directly.

    from agent import IncidentTriageAgent, build_reasoner

    agent = IncidentTriageAgent(kernel, build_reasoner(offline=True))
    result = agent.run("checkout p99 latency is 4s and climbing")
"""

from agent.reasoning import (
    BedrockReasoner,
    OfflineRuleReasoner,
    Reasoner,
    ReasoningError,
    build_reasoner,
)
from agent.triage import (
    AGENT_ID,
    TRIGGER_ROLE,
    IncidentTriageAgent,
    TriageResult,
    branch_name_for,
    retract_memory,
    triage,
)

__all__ = [
    "AGENT_ID",
    "TRIGGER_ROLE",
    "BedrockReasoner",
    "IncidentTriageAgent",
    "OfflineRuleReasoner",
    "Reasoner",
    "ReasoningError",
    "TriageResult",
    "branch_name_for",
    "build_reasoner",
    "retract_memory",
    "triage",
]

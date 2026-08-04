"""Tests for the Phase 6 demo agent and the kernel read paths it needs.

The behaviour worth pinning down here is not "does the agent call the kernel" —
it is the property the whole demo rests on: **the agent's decision is a function
of the memories it was given**, so withdrawing a memory changes the answer. If
that ever stops being true, the rewind contrast becomes theatre, and these tests
are what catches it.

Bedrock is never called. The reasoning path is exercised with an injected fake
client, and the semantic path with :class:`kernel.embeddings.FakeEmbeddingProvider`.
"""

from __future__ import annotations

import json
import uuid

import pytest

from agent.reasoning import (
    BedrockReasoner,
    OfflineRuleReasoner,
    ReasoningError,
    build_reasoner,
    format_context,
    parse_decision,
)
from agent.triage import IncidentTriageAgent, branch_name_for
from kernel.models import AgentDecision, Memory

# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def test_parse_decision_reads_a_bare_object():
    decision = parse_decision('{"action": "restart it", "rationale": "because"}')
    assert decision.action == "restart it"
    assert decision.rationale == "because"


def test_parse_decision_tolerates_a_code_fence_and_prose():
    # A model that wraps its answer has still answered; only a reply with no
    # object at all is a failure.
    reply = 'Sure!\n```json\n{"action": "scale up", "rationale": null}\n```\nHope that helps.'
    assert parse_decision(reply).action == "scale up"


def test_parse_decision_rejects_a_reply_with_no_json():
    with pytest.raises(ReasoningError, match="no JSON object"):
        parse_decision("I am not going to answer in JSON.")


def test_parse_decision_rejects_a_reply_with_no_action():
    with pytest.raises(ReasoningError, match="no 'action'"):
        parse_decision('{"rationale": "I thought about it"}')


def test_parse_decision_rejects_malformed_json():
    with pytest.raises(ReasoningError, match="malformed JSON"):
        parse_decision('{"action": "do it",,,}')


# ---------------------------------------------------------------------------
# Bedrock wiring (with an injected client — no AWS)
# ---------------------------------------------------------------------------


class _FakeBedrock:
    """Records the Converse call and replays a canned model reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self.reply}]}}}


def test_bedrock_reasoner_sends_incident_and_memories_and_parses_the_reply():
    client = _FakeBedrock('{"action": "raise max_client_conn", "rationale": "RB-021"}')
    reasoner = BedrockReasoner(
        model_id="anthropic.test-model", region="us-east-1", client=client
    )

    memory = Memory(
        id=uuid.uuid4(),
        branch_id=uuid.uuid4(),
        kind="runbook",
        content="RB-021: confirm max_client_conn before scaling",
        confidence=0.9,
        status="active",
        created_at="2026-01-01T00:00:00Z",
    )
    decision = reasoner.decide("latency spike", [memory])

    assert decision.action == "raise max_client_conn"
    assert reasoner.name == "bedrock:anthropic.test-model"

    sent = client.calls[0]
    assert sent["modelId"] == "anthropic.test-model"
    # The incident and the memory must both reach the model, or the decision
    # was not grounded in recalled memory at all.
    prompt = sent["messages"][0]["content"][0]["text"]
    assert "latency spike" in prompt
    assert "RB-021" in prompt


def test_bedrock_reasoner_wraps_client_errors_with_actionable_advice():
    from botocore.exceptions import ClientError

    class _Failing:
        def converse(self, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": "bad model id"}},
                "Converse",
            )

    reasoner = BedrockReasoner(
        model_id="anthropic.nope", region="us-east-1", client=_Failing()
    )
    with pytest.raises(ReasoningError) as excinfo:
        reasoner.decide("incident", [])
    message = str(excinfo.value)
    assert "BEDROCK_REASONING_MODEL" in message
    assert "anthropic.nope" in message


# ---------------------------------------------------------------------------
# The offline reasoner is memory-driven, not canned
# ---------------------------------------------------------------------------


def _memory(content: str, kind: str = "runbook") -> Memory:
    return Memory(
        id=uuid.uuid4(),
        branch_id=uuid.uuid4(),
        kind=kind,
        content=content,
        confidence=1.0,
        status="active",
        created_at="2026-01-01T00:00:00Z",
    )


RESTART_RUNBOOK = "RB-014: restart the pgbouncer pods to clear stuck connections"
RELOAD_RUNBOOK = (
    "RB-031: NEVER restart the pgbouncer pods during a live incident. "
    "Apply max_client_conn with a SIGHUP config reload."
)


def test_offline_reasoner_follows_the_restart_runbook_when_that_is_all_it_has():
    decision = OfflineRuleReasoner().decide("latency spike", [_memory(RESTART_RUNBOOK)])
    assert "restart" in decision.action.lower()


def test_offline_reasoner_changes_its_mind_when_the_runbook_is_replaced():
    """The property the entire rewind demo depends on.

    Same agent, same incident — only the memories differ. If this ever passes
    trivially (a canned answer), the side-by-side rewind proves nothing.
    """
    reasoner = OfflineRuleReasoner()
    before = reasoner.decide("latency spike", [_memory(RESTART_RUNBOOK)])
    after = reasoner.decide("latency spike", [_memory(RELOAD_RUNBOOK)])

    assert before.action != after.action
    assert "restart" in before.action.lower()
    assert "sighup" in after.action.lower()
    assert "do not restart" in after.action.lower()


def test_offline_reasoner_escalates_rather_than_inventing_a_remedy():
    decision = OfflineRuleReasoner().decide("something nobody has seen", [])
    assert "escalate" in decision.action.lower()


def test_offline_reasoner_is_deterministic():
    reasoner = OfflineRuleReasoner()
    memories = [_memory(RESTART_RUNBOOK)]
    assert reasoner.decide("x", memories) == reasoner.decide("x", memories)


def test_build_reasoner_selects_by_flag():
    assert isinstance(build_reasoner(offline=True), OfflineRuleReasoner)
    assert isinstance(build_reasoner(offline=False), BedrockReasoner)


def test_format_context_carries_similarity_when_present():
    from kernel.models import RecallResult

    hit = RecallResult(memory=_memory("a fact"), similarity=0.875, rank=1)
    rendered = format_context([hit])
    assert "similarity=0.875" in rendered
    # A bare Memory has no score, and must not grow a fake one.
    assert "similarity" not in format_context([_memory("a fact")])


def test_branch_name_is_slugged_and_stamped():
    name = branch_name_for("Payments p99 latency 4.2s!!")
    assert name.startswith("incident/payments-p99-latency-4-2s")
    assert "!" not in name


def test_branch_names_are_unique_within_the_same_second():
    """Regression: `branches.name` is uniquely indexed.

    A second-resolution timestamp alone collided when the same incident was
    fired twice inside one second (a double-click, or a retry), and the second
    run failed with a raw UniqueViolation.
    """
    from datetime import UTC, datetime

    at = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    names = {branch_name_for("identical incident text", at) for _ in range(200)}
    assert len(names) == 200


# ---------------------------------------------------------------------------
# End-to-end against a live cluster (skipped when unavailable, like the rest
# of the suite)
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(kernel):
    return IncidentTriageAgent(kernel, OfflineRuleReasoner(), k=5)


def test_triage_forks_records_and_captures_provenance(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook", source="runbooks/RB-014.md")
    kernel.remember("main", "pgbouncer max_client_conn is 200", "config")

    result = agent.run("pgbouncer latency spike, cl_waiting rising")

    # It forked rather than writing on main.
    assert result.branch.name.startswith("incident/")
    assert result.branch.parent_branch_id is not None
    assert result.decision.branch_id == result.branch.id

    # The incident itself was recorded on the fork, tagged so a replayed re-run
    # can find it again.
    assert result.incident_memory is not None
    assert result.incident_memory.branch_id == result.branch.id
    assert result.incident_memory.metadata["role"] == "incident-trigger"

    # Provenance captured similarity and rank as of decision time.
    from kernel import replay

    explanation = replay.explain_decision(kernel.db, kernel.actor, result.decision.id)
    assert len(explanation.memories) == len(result.recalled)
    assert all(m.similarity is not None for m in explanation.memories)
    assert explanation.has_invalidated_memories is False


def test_triage_leaves_main_untouched(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    before = len(kernel.list_memories("main", limit=100))

    agent.run("latency spike")

    assert len(kernel.list_memories("main", limit=100)) == before


def test_retracting_a_supporting_memory_makes_the_decision_suspect(kernel, agent):
    """explain_decision must flag a decision that rested on withdrawn memory."""
    from kernel import replay

    runbook = kernel.remember("main", RESTART_RUNBOOK, "runbook")
    result = agent.run("pgbouncer latency spike, cl_waiting rising")

    # Withdraw it on the decision's own branch — a fork froze main at its fork
    # point, so a retraction on main afterwards is (correctly) invisible here.
    kernel.retract(runbook.id, "caused INC-2350", branch=result.branch.id)

    explanation = replay.explain_decision(kernel.db, kernel.actor, result.decision.id)
    assert explanation.has_invalidated_memories is True
    assert explanation.invalidated_count >= 1
    assert runbook.id in explanation.invalidated_memory_ids
    flagged = next(m for m in explanation.memories if m.memory_id == runbook.id)
    assert flagged.retracted is True
    assert flagged.status_now == "retracted"


def test_rewind_faithful_matches_and_today_diverges(kernel, agent):
    """The demo's central claim, end to end.

    Faithful replay must reproduce the original action (proving the replay is
    honest); the re-run against today must diverge (proving the memory, not the
    agent, is what changed).
    """
    from kernel import replay

    runbook = kernel.remember("main", RESTART_RUNBOOK, "runbook")
    result = agent.run("pgbouncer latency spike, cl_waiting rising")
    assert "restart" in result.action.lower()

    # Learn better: withdraw the old runbook and record its replacement, both
    # on the branch that made the decision.
    kernel.retract(runbook.id, "caused INC-2350", branch=result.branch.id)
    kernel.remember(result.branch.id, RELOAD_RUNBOOK, "runbook")

    faithful = replay.rewind_and_rerun(
        kernel.db, kernel.actor, result.decision.id, agent.rerun_callable()
    )
    window = replay.replay_window_bounds(kernel.db)
    today = replay.rewind_and_rerun(
        kernel.db,
        kernel.actor,
        result.decision.id,
        agent.rerun_callable(),
        as_of=window.latest,
    )

    assert faithful.action_changed is False
    assert faithful.new_action == result.action

    assert today.action_changed is True
    assert "sighup" in today.new_action.lower()

    # And the availability diff explains why.
    assert len(today.memory_diff.only_then) >= 1
    assert len(today.memory_diff.only_now) >= 1


def test_triage_can_commit_the_branch_back(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    result = agent.run("latency spike", commit=True)
    assert result.committed is True
    assert result.commit_conflicts == 0


# ---------------------------------------------------------------------------
# Kernel read paths added for the console
# ---------------------------------------------------------------------------


def test_list_branches_reports_owned_counts_not_inherited(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    kernel.remember("main", "another fact", "fact")
    result = agent.run("latency spike")

    branches = {b.name: b for b in kernel.list_branches()}
    assert branches["main"].memory_count == 2
    assert branches["main"].child_count == 1
    assert branches["main"].parent_branch_id is None

    fork = branches[result.branch.name]
    # The fork owns only the incident memory it wrote — the two it inherited
    # are main's, and counting them here would double-count the corpus.
    assert fork.memory_count == 1
    assert fork.decision_count == 1


def test_list_decisions_scopes_to_a_branch(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    first = agent.run("latency spike one")
    agent.run("latency spike two")

    assert len(kernel.list_decisions()) == 2
    scoped = kernel.list_decisions(branch=first.branch.id)
    assert len(scoped) == 1
    assert scoped[0].id == first.decision.id


def test_list_decisions_is_newest_first(kernel, agent):
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    first = agent.run("incident one")
    second = agent.run("incident two")

    ids = [d.id for d in kernel.list_decisions()]
    assert ids.index(second.decision.id) < ids.index(first.decision.id)


# ---------------------------------------------------------------------------
# The rerun adapter reads its incident out of replayed state
# ---------------------------------------------------------------------------


def test_rerun_callable_uses_the_replayed_incident_not_a_smuggled_one(kernel):
    """The re-run must be driven only by what the branch contained.

    If the incident text were passed in from the present, a rewind would not be
    a reconstruction — it would be today's question asked of yesterday's data.
    """
    seen: dict[str, object] = {}

    class _Recording:
        name = "recording"

        def decide(self, incident, memories):
            seen["incident"] = incident
            seen["memories"] = list(memories)
            return AgentDecision(action="noted")

    agent = IncidentTriageAgent(kernel, _Recording())
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    result = agent.run("the original incident text")

    from kernel import replay

    replay.rewind_and_rerun(
        kernel.db, kernel.actor, result.decision.id, agent.rerun_callable()
    )

    assert seen["incident"] == "the original incident text"
    # The trigger memory is consumed as the prompt, not handed back as context.
    contents = [m.content for m in seen["memories"]]
    assert "the original incident text" not in contents
    assert any(RESTART_RUNBOOK in c for c in contents)


def test_json_round_trip_of_the_triage_summary(kernel, agent):
    """The API returns this verbatim, so it must be JSON-serialisable."""
    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    summary = agent.run("latency spike").summary()
    encoded = json.dumps(summary)
    assert json.loads(encoded)["action"] == summary["action"]
    assert "recalled" in summary and len(summary["recalled"]) >= 1


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_reasoning_failure_discards_the_speculative_branch(kernel):
    """A fork opened for work that never happened must not linger as open.

    Regression: a geo-restricted Bedrock model failed *after* the fork, leaving
    an open branch holding an incident and no decision — indistinguishable in
    the branch tree from triage still in progress. Observed against a live
    account.
    """

    class _Failing:
        name = "always-fails"

        def decide(self, incident, memories):
            raise ReasoningError("simulated model outage")

    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    agent = IncidentTriageAgent(kernel, _Failing())

    with pytest.raises(ReasoningError, match="simulated model outage"):
        agent.run("latency spike")

    forks = [b for b in kernel.list_branches() if b.name != "main"]
    assert len(forks) == 1
    assert forks[0].status == "discarded"
    # Discarded, never deleted — the incident memory is still readable.
    assert forks[0].memory_count == 1
    assert forks[0].decision_count == 0


def test_reasoning_failure_leaves_main_untouched(kernel):
    class _Failing:
        name = "always-fails"

        def decide(self, incident, memories):
            raise ReasoningError("boom")

    kernel.remember("main", RESTART_RUNBOOK, "runbook")
    before = len(kernel.list_memories("main", limit=100))

    with pytest.raises(ReasoningError):
        IncidentTriageAgent(kernel, _Failing()).run("latency spike")

    assert len(kernel.list_memories("main", limit=100)) == before


# ---------------------------------------------------------------------------
# Bedrock error diagnosis — the remedy must match the actual cause
# ---------------------------------------------------------------------------


def test_geo_restriction_is_not_reported_as_a_config_problem():
    """The geo block cannot be fixed by editing BEDROCK_REASONING_MODEL.

    Telling an operator to check their model id here sends them to change a
    setting that is already correct.
    """
    from agent.reasoning import _diagnose

    advice = _diagnose(
        "Access to Anthropic models is not allowed from unsupported countries, "
        "regions, or territories.",
        "us.anthropic.claude-sonnet-5",
    )
    assert "NOT a model-id or permissions problem" in advice
    assert "will not help" in advice
    assert "--offline" in advice


def test_bare_model_id_is_diagnosed_as_a_missing_routing_prefix():
    from agent.reasoning import _diagnose

    advice = _diagnose(
        "The provided model identifier is invalid.", "anthropic.claude-sonnet-5"
    )
    assert "INFERENCE_PROFILE-only" in advice
    assert "no routing prefix" in advice

    # A prefixed id that is still invalid should not blame the prefix.
    prefixed = _diagnose(
        "The provided model identifier is invalid.", "us.anthropic.claude-sonnet-5"
    )
    assert "no routing prefix" not in prefixed

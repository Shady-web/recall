"""The Recall MCP server, exercised through the MCP boundary itself.

Every test here calls ``FastMCP.call_tool`` rather than the kernel, because the
things Phase 5 has to guarantee — typed JSON out, read-only enforcement, MCP
actor tagging on audit rows — are properties of the *boundary*, and testing the
kernel underneath it would prove none of them.

Tools round-trip against a live CockroachDB instance (the suite skips cleanly
when none is reachable). Embeddings use :class:`FakeEmbeddingProvider`, so
nothing here needs AWS credentials or spends Bedrock quota.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest

from kernel.memory import MemoryKernel
from mcp_server.config import MCP_ACTOR_PREFIX, McpSettings
from mcp_server.server import READ_TOOLS, WRITE_TOOLS, build_server
from tests.conftest import audit_count, requires_crdb, table_count

pytestmark = requires_crdb

MCP_ACTOR = "mcp:pytest"

# The tool list Phase 5 contracts for, with whether each mutates state.
EXPECTED_TOOLS = {
    "remember": "write",
    "recall": "read",
    "branch": "write",
    "commit": "write",
    "discard": "write",
    "diff": "read",
    "explain_decision": "read",
    "rewind": "read",
}


# -- fixtures --------------------------------------------------------------


def _build(db, embedder, *, read_only: bool):
    kernel = MemoryKernel(
        db, actor=MCP_ACTOR, read_only=read_only, embedder=embedder
    )
    return build_server(
        kernel, McpSettings(actor=MCP_ACTOR, read_only=read_only, server_name="recall")
    )


@pytest.fixture
def server(db, fake_embedder):
    """A writable Recall MCP server bound to a throwaway database."""
    return _build(db, fake_embedder, read_only=False)


@pytest.fixture
def ro_server(db, fake_embedder):
    """The same server with ``RECALL_READ_ONLY`` in force."""
    return _build(db, fake_embedder, read_only=True)


async def call(server, tool: str, /, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool and return its structured JSON payload.

    ``server`` and ``tool`` are positional-only so that a tool argument can be
    named anything — the ``branch`` tool takes ``name``, which would otherwise
    collide with this helper's own parameters.

    ``FastMCP.call_tool`` returns ``(content_blocks, structured_content)``; the
    structured half is the contract these tests care about.
    """
    _content, structured = await server.call_tool(tool, arguments)
    return structured


def data(payload: dict[str, Any]) -> dict[str, Any]:
    """Assert a tool succeeded and return its ``data``."""
    assert payload["ok"] is True, payload
    return payload["data"]


def audit_actors(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT actor FROM audit_log")}


# -- the tool contract -----------------------------------------------------


async def test_exposes_exactly_the_contracted_tools(server):
    tools = await server.list_tools()
    assert {t.name for t in tools} == set(EXPECTED_TOOLS)


async def test_read_and_write_tools_are_annotated_correctly(server):
    for tool in await server.list_tools():
        expected_read_only = EXPECTED_TOOLS[tool.name] == "read"
        assert tool.annotations is not None, tool.name
        assert tool.annotations.readOnlyHint is expected_read_only, tool.name
    assert WRITE_TOOLS | READ_TOOLS == set(EXPECTED_TOOLS)


async def test_every_tool_declares_an_object_output_schema(server):
    # Structured JSON, not prose: each tool advertises an object result.
    for tool in await server.list_tools():
        assert tool.outputSchema is not None, tool.name
        assert tool.outputSchema["type"] == "object", tool.name


# -- round trips against a live cluster ------------------------------------


async def test_remember_and_recall_round_trip(server):
    stored = data(
        await call(
            server,
            "remember",
            branch="main",
            content="the payments database ran out of disk",
            kind="incident",
            source="pagerduty",
            confidence=0.9,
        )
    )["memory"]
    assert stored["kind"] == "incident"
    assert stored["source"] == "pagerduty"
    assert stored["confidence"] == 0.9

    hits = data(await call(server, "recall", branch="main", query="payments disk"))
    assert hits["count"] >= 1
    top = hits["results"][0]
    assert top["memory"]["id"] == stored["id"]
    assert top["rank"] == 1
    assert 0.0 <= top["similarity"] <= 1.0


async def test_recall_honours_structured_filters(server):
    await call(
        server, "remember", branch="main", content="disk pressure on host a", kind="incident"
    )
    await call(
        server, "remember", branch="main", content="disk pressure on host b", kind="note"
    )

    filtered = data(
        await call(
            server,
            "recall",
            branch="main",
            query="disk pressure",
            filters={"kind": "incident"},
        )
    )
    assert filtered["count"] == 1
    assert filtered["results"][0]["memory"]["kind"] == "incident"


async def test_branch_forks_and_inherits_parent_memories(server):
    await call(
        server, "remember", branch="main", content="baseline fact about servers", kind="fact"
    )
    child = data(await call(server, "branch", parent="main", name="experiment"))["branch"]
    assert child["name"] == "experiment"
    assert child["status"] == "open"
    assert child["parent_branch_id"] is not None

    inherited = data(await call(server, "recall", branch="experiment", query="servers"))
    assert any(
        "baseline fact" in r["memory"]["content"] for r in inherited["results"]
    )


async def test_branch_writes_do_not_leak_to_the_parent(server):
    await call(server, "branch", parent="main", name="experiment")
    await call(
        server,
        "remember",
        branch="experiment",
        content="speculative claim about servers",
        kind="fact",
    )
    parent = data(await call(server, "recall", branch="main", query="servers"))
    assert parent["count"] == 0


async def test_commit_round_trip(server):
    await call(server, "branch", parent="main", name="experiment")
    await call(
        server,
        "remember",
        branch="experiment",
        content="the fix was to raise the disk quota",
        kind="fact",
    )

    result = data(await call(server, "commit", branch="experiment"))
    assert result["committed"] is True
    assert result["conflict_count"] == 0
    assert len(result["replayed_memory_ids"]) == 1

    # The memory is now visible on the parent.
    merged = data(await call(server, "recall", branch="main", query="disk quota"))
    assert merged["count"] == 1


async def test_commit_returns_structured_conflicts_not_an_error(server, kernel):
    stored = data(
        await call(
            server,
            "remember",
            branch="main",
            content="the timeout is 30s on the server",
            kind="fact",
        )
    )["memory"]
    await call(server, "branch", parent="main", name="experiment")
    # Both sides change the same inherited memory after the fork point.
    kernel.supersede(stored["id"], "the timeout is 60s on the server", branch="experiment")
    kernel.supersede(stored["id"], "the timeout is 45s on the server")

    payload = await call(server, "commit", branch="experiment")
    # A conflict is an outcome, not a failure: ok stays true.
    result = data(payload)
    assert result["committed"] is False
    assert result["conflict_count"] == 1
    conflict = result["conflicts"][0]
    assert conflict["memory_id"] == stored["id"]
    assert conflict["branch_status"] == "superseded"
    assert conflict["reason"]
    # No-op: nothing replayed, branch left open for the caller to resolve.
    assert result["replayed_memory_ids"] == []


async def test_discard_round_trip(server):
    await call(server, "branch", parent="main", name="deadend")
    discarded = data(
        await call(server, "discard", branch="deadend", reason="hypothesis was wrong")
    )["branch"]
    assert discarded["status"] == "discarded"


async def test_diff_round_trip(server):
    await call(server, "branch", parent="main", name="left")
    await call(server, "branch", parent="main", name="right")
    left_memory = data(
        await call(server, "remember", branch="left", content="left side fact", kind="fact")
    )["memory"]

    result = data(await call(server, "diff", branch_a="left", branch_b="right"))
    assert result["a"]["name"] == "left"
    assert result["b"]["name"] == "right"
    assert left_memory["id"] in result["a"]["added"]
    assert result["b"]["added"] == []


async def _recorded_decision(server, kernel):
    """Store two memories over MCP, then record a decision citing them."""
    await call(
        server, "remember", branch="main", content="disk is full on host db-1", kind="incident"
    )
    await call(
        server,
        "remember",
        branch="main",
        content="restarting the ingester clears disk pressure",
        kind="runbook",
    )
    recalled = kernel.recall("main", "disk pressure", k=2)
    decision = kernel.record_decision(
        "main",
        agent_id="triage-agent",
        action="restart the ingester",
        rationale="disk pressure on db-1",
        recalled=recalled,
    )
    return decision, recalled


async def test_explain_decision_round_trip(server, kernel):
    decision, recalled = await _recorded_decision(server, kernel)

    result = data(await call(server, "explain_decision", decision_id=str(decision.id)))
    assert result["decision"]["id"] == str(decision.id)
    assert result["branch_name"] == "main"
    assert len(result["memories"]) == len(recalled)
    assert result["has_invalidated_memories"] is False
    first = result["memories"][0]
    assert first["rank"] == 1
    assert first["similarity"] is not None
    assert first["status_now"] == "active"


async def test_explain_decision_flags_memories_invalidated_since(server, kernel):
    decision, recalled = await _recorded_decision(server, kernel)
    retracted = recalled[0].memory.id
    kernel.retract(retracted, reason="the disk alert was a false positive")

    result = data(await call(server, "explain_decision", decision_id=str(decision.id)))
    assert result["has_invalidated_memories"] is True
    assert str(retracted) in result["invalidated_memory_ids"]


async def test_rewind_returns_a_logical_replay_summary(server, kernel, test_dsn):
    decision, recalled = await _recorded_decision(server, kernel)
    retracted = recalled[0].memory.id
    kernel.retract(retracted, reason="the disk alert was a false positive")

    result = data(await call(server, "rewind", decision_id=str(decision.id)))

    assert result["decision_id"] == str(decision.id)
    assert result["branch_name"] == "main"
    assert result["action"] == "restart the ingester"
    assert str(retracted) in result["contributing_memory_ids"]

    # The memory the agent relied on was available then and is gone now.
    diff = result["memory_diff"]
    assert diff["then_count"] == 2
    assert diff["now_count"] == 1
    assert [m["memory_id"] for m in diff["only_then"]] == [str(retracted)]
    assert {m["memory_id"] for m in result["memories_at_decision"]} == {
        str(r.memory.id) for r in recalled
    }


async def test_rewind_does_not_rerun_the_agent(server, kernel, test_dsn):
    decision, _ = await _recorded_decision(server, kernel)
    before = table_count(test_dsn, "decisions")

    result = data(await call(server, "rewind", decision_id=str(decision.id)))

    # No new decision was produced, and no re-run fields are reported.
    assert table_count(test_dsn, "decisions") == before
    assert "new_action" not in result
    assert "action_changed" not in result


# -- read-only mode --------------------------------------------------------


@pytest.mark.parametrize("tool", sorted(WRITE_TOOLS))
async def test_read_only_mode_refuses_every_write_tool(ro_server, tool, test_dsn):
    arguments = {
        "remember": {"branch": "main", "content": "should never land", "kind": "fact"},
        "branch": {"parent": "main", "name": "should-never-exist"},
        "commit": {"branch": "main"},
        "discard": {"branch": "main"},
    }[tool]

    payload = await call(ro_server, tool, **arguments)

    assert payload["ok"] is False
    assert payload["error"]["type"] == "read_only"
    assert "read-only" in payload["error"]["message"]
    assert payload["error"]["details"]["read_only"] is True
    # Refused at the MCP boundary: the database was never reached, so not even
    # an audit row exists for the attempt.
    assert audit_count(test_dsn) == 0


async def test_read_only_mode_still_serves_read_tools(ro_server, kernel):
    kernel.remember("main", "a fact written before read-only mode", kind="fact")

    hits = data(await call(ro_server, "recall", branch="main", query="read-only mode"))
    assert hits["count"] == 1
    assert data(await call(ro_server, "diff", branch_a="main", branch_b="main"))


async def test_read_only_mode_marks_write_tools_in_their_descriptions(ro_server):
    tools = {t.name: t for t in await ro_server.list_tools()}
    for name in WRITE_TOOLS:
        assert "read-only" in tools[name].description.lower(), name
    for name in READ_TOOLS:
        assert "read-only" not in tools[name].description.lower(), name


async def test_write_tools_are_still_listed_in_read_only_mode(ro_server):
    # Deliberate: a refusal an agent can read beats a capability that silently
    # vanishes from the tool list.
    assert {t.name for t in await ro_server.list_tools()} == set(EXPECTED_TOOLS)


# -- audit identity --------------------------------------------------------


async def test_audit_rows_carry_the_mcp_actor_tag(server, test_dsn):
    await call(server, "remember", branch="main", content="an audited fact", kind="fact")
    await call(server, "recall", branch="main", query="audited")
    await call(server, "branch", parent="main", name="audited-branch")
    await call(server, "diff", branch_a="main", branch_b="audited-branch")

    actors = audit_actors(test_dsn)
    assert actors == {MCP_ACTOR}
    assert all(a.startswith(MCP_ACTOR_PREFIX) for a in actors)


async def test_mcp_ops_are_distinguishable_from_direct_kernel_ops(server, kernel, test_dsn):
    kernel.remember("main", "written by a direct kernel caller", kind="fact")
    await call(server, "remember", branch="main", content="written over mcp", kind="fact")

    with psycopg.connect(test_dsn) as conn:
        rows = dict(
            conn.execute(
                "SELECT actor, count(*) FROM audit_log WHERE op = 'remember' "
                "GROUP BY actor"
            ).fetchall()
        )
    # The kernel fixture's actor is 'tester'; ours is prefixed.
    assert rows == {"tester": 1, MCP_ACTOR: 1}


async def test_every_tool_audits_under_the_mcp_actor(server, kernel, test_dsn):
    """No tool reaches the database on an unprefixed identity."""
    decision, _ = await _recorded_decision(server, kernel)
    await call(server, "branch", parent="main", name="audit-sweep")
    await call(server, "recall", branch="main", query="disk")
    await call(server, "diff", branch_a="main", branch_b="audit-sweep")
    await call(server, "explain_decision", decision_id=str(decision.id))
    await call(server, "rewind", decision_id=str(decision.id))
    await call(server, "commit", branch="audit-sweep")
    await call(server, "branch", parent="main", name="audit-sweep-two")
    await call(server, "discard", branch="audit-sweep-two")

    with psycopg.connect(test_dsn) as conn:
        stray = conn.execute(
            "SELECT DISTINCT actor, op FROM audit_log WHERE actor NOT LIKE %s",
            (f"{MCP_ACTOR_PREFIX}%",),
        ).fetchall()
    # Only the direct-kernel setup calls (actor 'tester') may be unprefixed.
    assert {actor for actor, _ in stray} <= {"tester"}


# -- typed errors ----------------------------------------------------------


async def test_unknown_branch_returns_typed_not_found(server):
    payload = await call(server, "recall", branch="no-such-branch", query="anything")
    assert payload["ok"] is False
    assert payload["error"]["type"] == "not_found"
    assert "no-such-branch" in payload["error"]["message"]


async def test_unknown_decision_returns_typed_not_found(server):
    payload = await call(
        server, "explain_decision", decision_id="00000000-0000-0000-0000-000000000000"
    )
    assert payload["ok"] is False
    assert payload["error"]["type"] == "not_found"


async def test_illegal_state_returns_typed_invalid_state(server):
    # 'main' is the root branch — there is nothing to commit into.
    payload = await call(server, "commit", branch="main")
    assert payload["ok"] is False
    assert payload["error"]["type"] == "invalid_state"


@pytest.mark.parametrize(
    "tool,arguments,field",
    [
        ("remember", {"branch": "main", "content": "", "kind": "fact"}, "content"),
        (
            "remember",
            {"branch": "main", "content": "x", "kind": "fact", "confidence": 5.0},
            "confidence",
        ),
        ("recall", {"branch": "main", "query": "q", "k": 0}, "k"),
        (
            "recall",
            {"branch": "main", "query": "q", "filters": {"min_confidance": 0.5}},
            "filters.min_confidance",
        ),
        ("branch", {"parent": "main", "name": "not a valid name!"}, "name"),
        ("explain_decision", {"decision_id": "not-a-uuid"}, "decision_id"),
        ("rewind", {"decision_id": "not-a-uuid"}, "decision_id"),
    ],
)
async def test_invalid_input_is_rejected_at_the_boundary(
    server, test_dsn, tool, arguments, field
):
    payload = await call(server, tool, **arguments)

    assert payload["ok"] is False
    assert payload["error"]["type"] == "invalid_input"
    assert field in {f["field"] for f in payload["error"]["details"]["fields"]}
    # Rejected before any kernel call, so nothing was audited and no embedding
    # was purchased.
    assert audit_count(test_dsn) == 0


async def test_errors_never_leak_a_traceback(server):
    payload = await call(server, "recall", branch="no-such-branch", query="anything")
    message = payload["error"]["message"]
    assert "Traceback" not in message
    assert ".py" not in message
    assert set(payload["error"]) == {"type", "message", "retryable", "details"}

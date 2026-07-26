"""The Recall MCP server — a thin wrapper that makes the kernel usable from
Claude Code, Cursor, and VS Code.

=============================================================================
WHAT THIS LAYER IS ALLOWED TO DO
=============================================================================

Almost nothing. Each tool does exactly four things, in order:

1. validate its arguments (:mod:`mcp_server.schemas`),
2. refuse the call if it is a write and the server is read-only,
3. call **one** kernel entry point,
4. serialize the kernel's return value to JSON.

There is no branching semantics, no SQL, no retry logic, and no state here —
all of that lives in ``kernel/`` and stays there. If a behaviour question can
be answered by reading this file, the answer is "whatever the kernel does".

=============================================================================
THREE PROPERTIES WORTH KNOWING
=============================================================================

**Audit parity.** Tools call the same kernel functions a direct Python caller
would, so every operation writes its audit row in the same transaction as the
operation itself — the MCP boundary adds no second path to the database. What
it *does* add is identity: the kernel is constructed with an actor carrying the
``mcp:`` prefix (see :mod:`mcp_server.config`), so ``audit_log`` distinguishes
an MCP-initiated fork from one made by a script or the demo agent.

**Read-only refuses rather than hides.** With ``RECALL_READ_ONLY`` set, the four
write tools stay listed but return a typed ``read_only`` error, and their
descriptions are prefixed so a model reading the tool list knows before calling.
Hiding them would be simpler, but an agent that cannot see a tool concludes the
capability does not exist and silently works around it; one that gets a clear
refusal reports the real reason. The refusal is enforced here *and* again in the
kernel — an MCP client cannot reach a write path through either door.

**Blocking work leaves the event loop.** Kernel calls are synchronous psycopg
(and, on ``remember``, a network embedding call). Each tool is ``async`` and
dispatches the kernel call through ``anyio.to_thread``, so a slow query cannot
stall the MCP session's heartbeat.

``rewind`` deserves one note: it is deliberately the *summary* path
(:func:`kernel.replay.rewind_summary`), not ``rewind_and_rerun``. A tool call
must never fire an agent — and therefore a model call — as a side effect of what
the caller asked to be a read.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Annotated, Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from kernel import replay as replay_mod
from kernel.db import Database, get_default_database
from kernel.embeddings import EmbeddingProvider
from kernel.memory import MemoryKernel
from mcp_server.config import McpSettings, load_mcp_settings
from mcp_server.errors import ErrorType, build_error, error_payload, ok_payload
from mcp_server.schemas import (
    BranchRequest,
    CommitRequest,
    DecisionRequest,
    DiffRequest,
    DiscardRequest,
    RecallRequest,
    RememberRequest,
)

logger = logging.getLogger("recall.mcp.server")

#: Tools that mutate state. Blocked at this boundary in read-only mode.
WRITE_TOOLS = frozenset({"remember", "branch", "commit", "discard"})

#: Tools that only read. Always available, including in read-only mode.
READ_TOOLS = frozenset({"recall", "diff", "explain_decision", "rewind"})

_READ_ONLY_NOTE = "[DISABLED — server is in read-only mode] "

SERVER_INSTRUCTIONS = """\
Recall is branchable, replayable agent memory backed by CockroachDB.

Use `remember` to store a durable fact and `recall` to retrieve facts by meaning,
always scoped to a branch (`main` is the root). Before speculating — trying a
risky fix, exploring a hypothesis — call `branch` to fork memory, work on the
fork, then `commit` to fold it into the parent or `discard` to throw it away.
Nothing is ever hard-deleted.

`explain_decision` reports which memories drove a recorded decision and flags any
that have since been superseded or retracted. `rewind` reconstructs what a branch
knew at the moment of a decision and diffs it against what it knows now.

Every tool returns JSON: {"ok": true, "data": ...} or
{"ok": false, "error": {"type": ..., "message": ...}}. Check `ok` before reading
`data`. A `commit` that finds conflicts returns ok=true with committed=false and
the conflicts attached — that is an outcome to resolve, not a failure.\
"""


def _json(model: Any) -> dict[str, Any]:
    """Serialize a kernel pydantic model to JSON-safe primitives."""
    return model.model_dump(mode="json")


class RecallTools:
    """Bound kernel handles plus the read-only policy for one server process.

    Holds no state of its own beyond configuration — every call goes straight to
    the kernel.
    """

    def __init__(self, kernel: MemoryKernel, settings: McpSettings) -> None:
        self.kernel = kernel
        self.settings = settings

    @property
    def db(self) -> Database:
        return self.kernel.db

    # -- dispatch ---------------------------------------------------------

    def _refusal(self, tool: str) -> dict[str, Any]:
        return build_error(
            tool,
            ErrorType.READ_ONLY,
            f"the Recall MCP server is running in read-only mode "
            f"(RECALL_READ_ONLY), so the write tool {tool!r} is disabled. "
            f"Read tools ({', '.join(sorted(READ_TOOLS))}) remain available. "
            f"Restart the server with RECALL_READ_ONLY=false to enable writes.",
            details={"read_only": True, "disabled_tools": sorted(WRITE_TOOLS)},
        )

    async def invoke(
        self, tool: str, work: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """Run one kernel call off the event loop and envelope its outcome.

        This is the only place a tool result or an exception becomes a wire
        payload, which is what guarantees no tool can leak a traceback.
        """
        if tool in WRITE_TOOLS and self.settings.read_only:
            return self._refusal(tool)
        try:
            data = await anyio.to_thread.run_sync(work)
        except Exception as exc:
            # Deliberately broad: a tool must never propagate an exception to the
            # transport, because the SDK would render it as a bare error string.
            return error_payload(tool, exc)
        return ok_payload(tool, data)

    # -- tool bodies (each: validate -> one kernel call -> serialize) ------

    def _remember(self, **kwargs: Any) -> dict[str, Any]:
        req = RememberRequest(**kwargs)
        memory = self.kernel.remember(
            req.branch,
            req.content,
            req.kind,
            source=req.source,
            confidence=req.confidence,
        )
        return {"memory": _json(memory)}

    def _recall(self, **kwargs: Any) -> dict[str, Any]:
        req = RecallRequest(**kwargs)
        f = req.filters
        results = self.kernel.recall(
            req.branch,
            req.query,
            k=req.k,
            kind=f.kind,
            min_confidence=f.min_confidence,
            since=f.since,
            status=f.status,
        )
        return {
            "count": len(results),
            "results": [_json(r) for r in results],
        }

    def _branch(self, **kwargs: Any) -> dict[str, Any]:
        req = BranchRequest(**kwargs)
        return {"branch": _json(self.kernel.fork(req.parent, req.name))}

    def _commit(self, **kwargs: Any) -> dict[str, Any]:
        req = CommitRequest(**kwargs)
        result = self.kernel.commit_branch(req.branch)
        payload = _json(result)
        # Surfaced at the top level so a caller can branch on the outcome
        # without walking the conflict list.
        payload["conflict_count"] = len(result.conflicts)
        return payload

    def _discard(self, **kwargs: Any) -> dict[str, Any]:
        req = DiscardRequest(**kwargs)
        return {"branch": _json(self.kernel.discard_branch(req.branch, req.reason))}

    def _diff(self, **kwargs: Any) -> dict[str, Any]:
        req = DiffRequest(**kwargs)
        return _json(self.kernel.diff_branches(req.branch_a, req.branch_b))

    def _explain_decision(self, **kwargs: Any) -> dict[str, Any]:
        req = DecisionRequest(**kwargs)
        return _json(
            replay_mod.explain_decision(self.db, self.kernel.actor, req.decision_id)
        )

    def _rewind(self, **kwargs: Any) -> dict[str, Any]:
        req = DecisionRequest(**kwargs)
        return _json(
            replay_mod.rewind_summary(self.db, self.kernel.actor, req.decision_id)
        )


def _describe(tool: str, description: str, read_only_server: bool) -> str:
    """Prefix a write tool's description when the server refuses writes."""
    if read_only_server and tool in WRITE_TOOLS:
        return _READ_ONLY_NOTE + description
    return description


def build_server(
    kernel: MemoryKernel,
    settings: McpSettings | None = None,
) -> FastMCP:
    """Construct a :class:`FastMCP` server exposing the Recall kernel.

    ``kernel`` is injected rather than constructed here so tests can supply a
    kernel bound to a throwaway database and a fake embedding provider.
    """
    settings = settings or load_mcp_settings()
    tools = RecallTools(kernel, settings)
    mcp: FastMCP = FastMCP(
        name=settings.server_name, instructions=SERVER_INSTRUCTIONS
    )

    def register(
        name: str,
        handler: Callable[..., Any],
        description: str,
        *,
        destructive: bool = False,
        idempotent: bool = False,
    ) -> None:
        """Register one tool, deriving its input schema from ``handler``'s signature."""
        mcp.add_tool(
            handler,
            name=name,
            description=_describe(name, description, settings.read_only),
            annotations=ToolAnnotations(
                readOnlyHint=name in READ_TOOLS,
                destructiveHint=destructive,
                idempotentHint=idempotent,
                openWorldHint=False,
            ),
        )

    # Each handler below is signature-explicit on purpose: FastMCP derives the
    # tool's JSON input schema from these annotations, so the signature *is* the
    # public contract. Value constraints stay in mcp_server.schemas, where a
    # violation can be reported as a typed error rather than a protocol fault.

    async def remember(
        branch: Annotated[str, Field(description="Branch name or id to store on.")],
        content: Annotated[str, Field(description="The fact to remember, as text.")],
        kind: Annotated[
            str,
            Field(description="Short category, e.g. 'fact', 'incident', 'preference'."),
        ],
        source: Annotated[
            str | None, Field(description="Where this fact came from.")
        ] = None,
        confidence: Annotated[
            float, Field(description="How trusted this fact is, 0.0-1.0.")
        ] = 1.0,
    ) -> dict[str, Any]:
        return await tools.invoke(
            "remember",
            functools.partial(
                tools._remember,
                branch=branch,
                content=content,
                kind=kind,
                source=source,
                confidence=confidence,
            ),
        )

    async def recall(
        branch: Annotated[str, Field(description="Branch name or id to search.")],
        query: Annotated[str, Field(description="What to look for, in natural language.")],
        k: Annotated[int, Field(description="Maximum hits to return, 1-100.")] = 10,
        filters: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Optional structured filters. Keys: kind (string), "
                    "min_confidence (0.0-1.0), since (ISO-8601 timestamp), status "
                    "(active|superseded|retracted, default 'active'; null for any). "
                    "Unknown keys are rejected."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        return await tools.invoke(
            "recall",
            functools.partial(
                tools._recall, branch=branch, query=query, k=k, filters=filters
            ),
        )

    async def branch(
        parent: Annotated[str, Field(description="Branch name or id to fork from.")],
        name: Annotated[str, Field(description="Name for the new branch.")],
    ) -> dict[str, Any]:
        return await tools.invoke(
            "branch", functools.partial(tools._branch, parent=parent, name=name)
        )

    async def commit(
        branch: Annotated[str, Field(description="Branch name or id to commit.")],
    ) -> dict[str, Any]:
        return await tools.invoke(
            "commit", functools.partial(tools._commit, branch=branch)
        )

    async def discard(
        branch: Annotated[str, Field(description="Branch name or id to discard.")],
        reason: Annotated[str, Field(description="Why it is being abandoned.")] = "",
    ) -> dict[str, Any]:
        return await tools.invoke(
            "discard", functools.partial(tools._discard, branch=branch, reason=reason)
        )

    async def diff(
        branch_a: Annotated[str, Field(description="First branch name or id.")],
        branch_b: Annotated[str, Field(description="Second branch name or id.")],
    ) -> dict[str, Any]:
        return await tools.invoke(
            "diff",
            functools.partial(tools._diff, branch_a=branch_a, branch_b=branch_b),
        )

    async def explain_decision(
        decision_id: Annotated[str, Field(description="UUID of the recorded decision.")],
    ) -> dict[str, Any]:
        return await tools.invoke(
            "explain_decision",
            functools.partial(tools._explain_decision, decision_id=decision_id),
        )

    async def rewind(
        decision_id: Annotated[str, Field(description="UUID of the recorded decision.")],
    ) -> dict[str, Any]:
        return await tools.invoke(
            "rewind", functools.partial(tools._rewind, decision_id=decision_id)
        )

    register(
        "remember",
        remember,
        "Store one atomic fact on a branch. The content is embedded on write, so "
        "it becomes retrievable by meaning via `recall`. Returns the stored memory "
        "including its id.",
    )
    register(
        "recall",
        recall,
        "Retrieve the memories most similar in meaning to a query, scoped to a "
        "branch and everything that branch inherited from its ancestors. Returns "
        "each hit with its similarity score and rank.",
        idempotent=True,
    )
    register(
        "branch",
        branch,
        "Fork a branch so work can proceed speculatively without touching the "
        "parent. The fork starts empty and inherits everything the parent knew at "
        "the fork point. Use this before trying anything you might want to undo.",
    )
    register(
        "commit",
        commit,
        "Fold a branch's memories into its parent and close the branch. If any "
        "memory was modified on both the branch and the parent since the fork, the "
        "commit is a NO-OP: it returns ok=true with committed=false, "
        "conflict_count>0, and a structured conflicts list, leaving the branch "
        "open for you to resolve.",
    )
    register(
        "discard",
        discard,
        "Abandon a branch, throwing away its speculative work. Nothing is "
        "hard-deleted — the branch is marked discarded and stays readable for "
        "audit and replay.",
        destructive=True,
    )
    register(
        "diff",
        diff,
        "Compare two branches. Returns, for each side, the memory ids it added, "
        "superseded, and retracted.",
        idempotent=True,
    )
    register(
        "explain_decision",
        explain_decision,
        "Explain why an agent decided what it did. Returns the decision plus every "
        "memory that drove it, each with the similarity and rank recorded at "
        "decision time and its status today — and flags any decision resting on "
        "memory since superseded or retracted.",
        idempotent=True,
    )
    register(
        "rewind",
        rewind,
        "Reconstruct what a branch knew at the moment of a decision and diff it "
        "against what it knows now. This is a logical replay summary only — it "
        "reports the memory state and what changed, and does NOT re-run the agent "
        "or produce a new decision.",
        idempotent=True,
    )
    return mcp


def create_server(
    db: Database | None = None,
    embedder: EmbeddingProvider | None = None,
) -> FastMCP:
    """Build the server from process configuration (the production path).

    The kernel is constructed with the MCP actor identity, so every audit row
    this process writes is attributable to MCP rather than to a bare script.
    """
    settings = load_mcp_settings()
    from kernel.embeddings import BedrockEmbeddingProvider

    kernel = MemoryKernel(
        db or get_default_database(),
        actor=settings.actor,
        read_only=settings.read_only,
        embedder=embedder or BedrockEmbeddingProvider(),
    )
    logger.info(
        "Recall MCP server ready (actor=%s, read_only=%s)",
        settings.actor,
        settings.read_only,
    )
    return build_server(kernel, settings)


def main() -> None:
    """Entry point: serve over stdio, the transport MCP clients spawn locally."""
    logging.basicConfig(level=logging.INFO)
    create_server().run(transport="stdio")

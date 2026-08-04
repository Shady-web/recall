"""Thin HTTP bridge between the web UI and the Recall kernel.

Deliberately thin, in the same sense as ``mcp_server/``: each endpoint validates
its arguments, calls **one** kernel entry point, and serializes the result. No
SQL, no branching semantics, no caching, no derived statistics. If a number
appears in a response it came out of CockroachDB on that request — which is what
lets the UI promise that everything on screen is live.

Two endpoints are not pure pass-throughs, and both say so where they are defined:

* ``POST /api/incidents`` runs the demo agent, which is a write plus a model call.
* ``POST /api/decisions/{id}/rerun`` runs :func:`kernel.replay.rewind_and_rerun`
  **twice** — once faithfully at decision time, once against today — because the
  contrast between those two runs is the thing the UI exists to show.

Run it with::

    uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import IncidentTriageAgent, build_reasoner
from api.events import HEARTBEAT_SECONDS, EventBroker, sse
from kernel import replay as replay_mod
from kernel.errors import (
    InvalidStateError,
    NotFoundError,
    ReadOnlyError,
    RecallError,
    ReplayWindowExpiredError,
)
from kernel.memory import MemoryKernel

logger = logging.getLogger("recall.api")

#: Kernel errors mapped to HTTP status. Anything unlisted is a 500, which is
#: correct: an unmapped kernel error is a bug here, not a client mistake.
_STATUS_FOR_ERROR: list[tuple[type[Exception], int]] = [
    (NotFoundError, 404),
    (ReadOnlyError, 403),
    (ReplayWindowExpiredError, 422),
    (InvalidStateError, 409),
]


def _http_error(exc: RecallError) -> HTTPException:
    for error_type, status in _STATUS_FOR_ERROR:
        if isinstance(exc, error_type):
            return HTTPException(
                status_code=status,
                detail={"type": type(exc).__name__, "message": str(exc)},
            )
    return HTTPException(
        status_code=500, detail={"type": type(exc).__name__, "message": str(exc)}
    )


class AppState:
    """Process-wide handles. Built once at startup, shared by every request."""

    kernel: MemoryKernel
    broker: EventBroker
    offline: bool
    dsn: str

    def agent(self) -> IncidentTriageAgent:
        return IncidentTriageAgent(self.kernel, build_reasoner(offline=self.offline))


state = AppState()


def _json(model: Any) -> Any:
    return model.model_dump(mode="json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from kernel.config import settings
    from kernel.db import Database, verify_embedding_dimension, verify_embedding_provider
    from kernel.embeddings import BedrockEmbeddingProvider, FakeEmbeddingProvider

    # RECALL_OFFLINE is how run_demo.sh keeps iteration free: fake embeddings and
    # the rule reasoner. It is surfaced on /api/health so the UI can label it.
    state.offline = os.environ.get("RECALL_OFFLINE", "").lower() in {"1", "true", "yes"}
    state.dsn = os.environ.get("RECALL_DSN") or settings.crdb_connection_string

    db = Database(state.dsn)
    embedder = FakeEmbeddingProvider() if state.offline else BedrockEmbeddingProvider()
    state.kernel = MemoryKernel(
        db,
        actor=settings.recall_actor_id,
        read_only=settings.recall_read_only,
        embedder=embedder,
    )
    verify_embedding_dimension(db, embedder)
    # Catches a fake-seeded database being served with real Bedrock, which
    # otherwise yields plausible-looking hits scored on orthogonal noise.
    verify_embedding_provider(db, embedder)

    state.broker = EventBroker(state.dsn)
    state.broker.start(asyncio.get_running_loop())
    logger.info("Recall API ready (offline=%s, actor=%s)", state.offline, state.kernel.actor)
    try:
        yield
    finally:
        state.broker.stop()
        db.close()


app = FastAPI(title="Recall API", version="0.6.0", lifespan=lifespan)

# The UI is served by Vite on another port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Health and cluster facts
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Live cluster and process facts. The UI's status bar reads only this."""
    kernel = state.kernel
    reachable = kernel.db.health_check()
    version = None
    if reachable:
        with kernel.db.transaction(read_only=True) as conn:
            version = conn.execute("SELECT version()").fetchone()[0]
    return {
        "cluster_reachable": reachable,
        "cluster_version": version,
        "actor": kernel.actor,
        "read_only": kernel.read_only,
        "offline_mode": state.offline,
        "embedding_provider": type(kernel.embedder).__name__,
        "reasoning_provider": build_reasoner(offline=state.offline).name,
        "live_feed": state.broker.status(),
    }


@app.get("/api/replay-window")
def replay_window() -> dict[str, Any]:
    """Bounds of PHYSICAL (``AS OF SYSTEM TIME``) replay, read live from the cluster.

    The UI greys out the forensic-replay control outside this range instead of
    letting the request fail — the limit is real and worth showing, not worth
    hiding. Logical replay is not bounded by it.
    """
    return _json(replay_mod.replay_window_bounds(state.kernel.db))


# --------------------------------------------------------------------------
# Branches
# --------------------------------------------------------------------------


@app.get("/api/branches")
def list_branches() -> dict[str, Any]:
    """Every branch with parent, fork point, status, and owned-row counts."""
    branches = state.kernel.list_branches()
    return {"count": len(branches), "branches": [_json(b) for b in branches]}


# Branch-scoped reads take the branch as a QUERY parameter, not a path segment.
# Branch names are free-form and the agent's are hierarchical
# (``incident/<slug>-<stamp>``); a name containing ``/`` cannot survive a path
# parameter even percent-encoded, because the ASGI server decodes ``%2F`` before
# routing and the request lands on a different (non-existent) route. A query
# parameter accepts any name, and ids work equally well.


@app.get("/api/memories")
def branch_memories(
    branch: str = Query(..., description="Branch name or id"),
    limit: int = Query(200, ge=1, le=1000),
    status: str | None = Query(None),
) -> dict[str, Any]:
    """Memories **owned by** this branch (not what it inherits).

    Use ``/api/replay`` for the resolved, ancestry-aware view.
    """
    try:
        memories = state.kernel.list_memories(branch, status=status, limit=limit)
    except RecallError as exc:
        raise _http_error(exc) from exc
    return {"count": len(memories), "memories": [_json(m) for m in memories]}


@app.get("/api/timeline")
def branch_timeline(
    branch: str = Query(..., description="Branch name or id"),
) -> dict[str, Any]:
    """The instants the scrubber can travel between, for this branch.

    ``earliest`` is the oldest memory visible from the branch (so the scrubber
    starts before the branch knew anything) and ``latest`` is cluster now.
    ``fork_point_ts`` is returned so the UI can mark where the branch began.
    """
    kernel = state.kernel
    try:
        window = replay_mod.replay_window_bounds(kernel.db)
        segments = kernel.ancestry(branch)
        visible = replay_mod.replay_branch_at(
            kernel.db, kernel.actor, branch, window.latest, status=None
        )
        branch_row = next(
            (b for b in kernel.list_branches() if str(b.id) == str(segments[0].branch_id)),
            None,
        )
    except RecallError as exc:
        raise _http_error(exc) from exc

    created = sorted(m.created_at for m in visible) if visible else []
    return {
        "branch": _json(branch_row) if branch_row is not None else None,
        "earliest": created[0].isoformat() if created else window.earliest.isoformat(),
        "latest": window.latest.isoformat(),
        "fork_point_ts": (
            branch_row.fork_point_ts.isoformat()
            if branch_row is not None and branch_row.fork_point_ts is not None
            else None
        ),
        "memory_count": len(visible),
        "ancestry": [_json(s) for s in segments],
        "replay_window": _json(window),
    }


@app.get("/api/replay")
def branch_replay(
    branch: str = Query(..., description="Branch name or id"),
    t: datetime = Query(..., description="ISO-8601 instant to reconstruct"),
    mode: str = Query("logical", pattern="^(logical|physical)$"),
    status: str | None = Query("active"),
) -> dict[str, Any]:
    """What ``branch`` knew at ``t``.

    ``mode=logical`` reconstructs from the validity columns and works at any
    age. ``mode=physical`` uses ``AS OF SYSTEM TIME`` for true historical bytes
    and is bounded by MVCC garbage collection — outside that window it returns
    **422** with the window it *can* serve, rather than empty or wrong data.
    """
    kernel = state.kernel
    try:
        if mode == "physical":
            memories = replay_mod.replay_cluster_at(
                kernel.db, kernel.actor, t, branch=branch
            )
        else:
            memories = replay_mod.replay_branch_at(
                kernel.db, kernel.actor, branch, t, status=status
            )
    except RecallError as exc:
        raise _http_error(exc) from exc
    return {
        "branch": branch,
        "at": t.isoformat(),
        "mode": mode,
        "count": len(memories),
        "memories": [_json(m) for m in memories],
    }


@app.post("/api/commit")
def commit_branch(branch: str = Body(..., embed=True)) -> dict[str, Any]:
    """Fold a branch into its parent.

    A conflicting commit is an outcome, not an error: it returns 200 with
    ``committed=false`` and the conflicts attached, exactly as the kernel
    reports it.
    """
    try:
        result = state.kernel.commit_branch(branch)
    except RecallError as exc:
        raise _http_error(exc) from exc
    payload = _json(result)
    payload["conflict_count"] = len(result.conflicts)
    return payload


# --------------------------------------------------------------------------
# Decisions and provenance
# --------------------------------------------------------------------------


@app.get("/api/decisions")
def list_decisions(
    branch: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Recorded decisions, newest first, optionally scoped to one branch."""
    try:
        decisions = state.kernel.list_decisions(branch=branch, limit=limit)
    except RecallError as exc:
        raise _http_error(exc) from exc
    return {"count": len(decisions), "decisions": [_json(d) for d in decisions]}


@app.get("/api/decisions/{decision_id}/explain")
def explain_decision(decision_id: str) -> dict[str, Any]:
    """Which memories drove a decision, and what has changed since.

    Each contributing memory carries the similarity and rank recorded **at
    decision time** plus its status **today**. ``has_invalidated_memories`` and
    ``invalidated_count`` are top-level so the UI can badge a suspect decision
    without walking the list.
    """
    try:
        return _json(
            replay_mod.explain_decision(state.kernel.db, state.kernel.actor, decision_id)
        )
    except RecallError as exc:
        raise _http_error(exc) from exc


@app.get("/api/decisions/{decision_id}/rewind")
def rewind_decision(decision_id: str) -> dict[str, Any]:
    """Read-only rewind: what the branch knew then vs now, **no agent run**.

    This is the safe one to call on selection. The agent only runs when the
    operator explicitly asks, via ``POST .../rerun``.
    """
    try:
        return _json(
            replay_mod.rewind_summary(state.kernel.db, state.kernel.actor, decision_id)
        )
    except RecallError as exc:
        raise _http_error(exc) from exc


@app.post("/api/decisions/{decision_id}/rerun")
def rerun_decision(decision_id: str) -> dict[str, Any]:
    """Run the agent **twice** against the same decision and return both runs.

    * ``faithful`` — replayed at the decision's own timestamp. A deterministic
      agent reproduces the original action; that it does is the fidelity check.
    * ``today`` — replayed against current state. If a supporting memory has
      since been retracted or superseded, this is where the action changes.

    Returning them together is the point: the pair is the evidence, and either
    half alone proves nothing. Both are real agent invocations — on a live
    (non-offline) deployment this costs two model calls.
    """
    kernel = state.kernel
    agent = state.agent()
    try:
        window = replay_mod.replay_window_bounds(kernel.db)
        faithful = replay_mod.rewind_and_rerun(
            kernel.db, kernel.actor, decision_id, agent.rerun_callable()
        )
        today = replay_mod.rewind_and_rerun(
            kernel.db,
            kernel.actor,
            decision_id,
            agent.rerun_callable(),
            as_of=window.latest,
        )
    except RecallError as exc:
        raise _http_error(exc) from exc
    return {
        "decision_id": decision_id,
        "reasoner": agent.reasoner.name,
        "faithful": _json(faithful),
        "today": _json(today),
        "verdict_changed": bool(today.action_changed),
    }


# --------------------------------------------------------------------------
# Writes (the demo's interactive beats)
# --------------------------------------------------------------------------


@app.post("/api/incidents")
def fire_incident(
    incident: str = Body(..., embed=True),
    commit: bool = Body(False, embed=True),
    base_branch: str = Body("main", embed=True),
) -> dict[str, Any]:
    """Fire an incident: recall, fork, reason, record provenance, optionally commit.

    This is a write path and, off ``RECALL_OFFLINE``, a real Bedrock call. Every
    row it creates arrives on the live feed while the request is still running,
    which is how new memories appear on screen mid-demo.
    """
    try:
        result = state.agent().run(incident, commit=commit, branch_name=None)
    except RecallError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:  # reasoning backends raise their own error type
        raise HTTPException(
            status_code=502, detail={"type": type(exc).__name__, "message": str(exc)}
        ) from exc
    return result.summary()


@app.post("/api/memories/{memory_id}/retract")
def retract_memory(
    memory_id: str,
    reason: str = Body(..., embed=True),
    branch: str | None = Body(None, embed=True),
) -> dict[str, Any]:
    """Withdraw a memory that turned out to be wrong.

    ``branch`` selects who is withdrawing it: omitted, the memory's own branch
    (which changes the row); a descendant branch, and the ancestor's row is left
    untouched while that branch alone stops seeing it.
    """
    try:
        return _json(state.kernel.retract(memory_id, reason, branch=branch))
    except RecallError as exc:
        raise _http_error(exc) from exc


# --------------------------------------------------------------------------
# Live feed
# --------------------------------------------------------------------------


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    """Server-Sent Events carrying cluster changes as they happen.

    The first frame is always ``hello``, whose ``mode`` is ``changefeed`` or
    ``poll`` — the UI displays that value rather than assuming push worked.
    """

    async def stream():
        with state.broker.subscribe() as queue:
            hello = {
                "type": "hello",
                "at": None,
                "data": state.broker.status(),
            }
            yield sse(hello)
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

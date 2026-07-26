"""End-to-end demo of Recall's replay and decision provenance.

Shows: an agent records a decision -> a supporting memory is later retracted ->
explain_decision flags the decision as resting on invalidated memory ->
rewind_and_rerun proves the agent would now decide differently.

Also demonstrates the logical/physical replay distinction and the GC-window
guard. Runs against a local CockroachDB with the deterministic fake embedder,
so it needs no AWS credentials:

    docker run -d --name recall-crdb -p 26257:26257 \\
        cockroachdb/cockroach:latest-v25.2 start-single-node --insecure
    python scripts/demo_replay.py
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from urllib.parse import urlparse, urlunparse

import psycopg

from kernel.db import Database
from kernel.embeddings import FakeEmbeddingProvider
from kernel.errors import ReplayWindowExpiredError
from kernel.memory import MemoryKernel
from kernel.migrate import migrate
from kernel.replay import (
    explain_decision,
    replay_branch_at,
    replay_cluster_at,
    replay_window_bounds,
    rewind_and_rerun,
)

BASE = os.environ.get(
    "RECALL_TEST_DSN", "postgresql://root@localhost:26257/defaultdb?sslmode=disable"
)


def _with_db(dsn: str, name: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{name}"))


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def triage_agent(ctx):
    """A tiny incident-triage agent. Decides purely from the memories it sees."""
    if any("healthy" in m.content for m in ctx.memories):
        return {"action": "no-op", "rationale": "the database reported healthy"}
    return {
        "action": "page-oncall",
        "rationale": "no evidence the database is healthy, and latency is elevated",
    }


def main() -> None:
    dbname = f"recall_replay_{uuid.uuid4().hex[:8]}"
    admin = _with_db(BASE, "defaultdb")
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
        c.execute(f"CREATE DATABASE {dbname}")
    dsn = _with_db(BASE, dbname)
    migrate(dsn=dsn)

    db = Database(dsn)
    k = MemoryKernel(db, actor="demo", read_only=False, embedder=FakeEmbeddingProvider())

    try:
        hr("1. The agent's world at incident time")
        healthy = k.remember("main", "the database is healthy", kind="fact",
                             source="healthcheck")
        k.remember("main", "checkout latency is elevated", kind="metric", source="monitor")
        k.remember("main", "a deploy landed 10 minutes ago", kind="event", source="ci")
        for m in k.list_memories("main"):
            print(f"    {m.kind:7} {m.content}")

        hr("2. The agent recalls, then decides")
        hits = k.recall("main", "database health", k=3)
        for h in hits:
            print(f"    #{h.rank} sim={h.similarity:+.3f}  {h.memory.content}")
        decision = k.record_decision(
            "main",
            agent_id="triage-agent",
            action="no-op",
            rationale="the database reported healthy",
            recalled=hits,
        )
        print(f"\n    DECISION: {decision.action!r} -- {decision.rationale}")

        hr("3. Later, we learn one supporting memory was wrong")
        k.retract(healthy.id, reason="healthcheck was reporting a stale cached value")
        print("    retracted: 'the database is healthy'")

        hr("4. explain_decision -- what did this decision rest on?")
        ex = explain_decision(db, "demo", decision.id)
        print(f"    decision {str(ex.decision.id)[:8]} on branch {ex.branch_name!r}")
        print(f"    action={ex.decision.action!r}")
        print(f"\n    *** has_invalidated_memories = {ex.has_invalidated_memories} "
              f"({ex.invalidated_count} of {len(ex.memories)}) ***\n")
        for c in ex.memories:
            flag = ""
            if c.retracted:
                flag = "  <-- RETRACTED SINCE THE DECISION"
            elif c.superseded:
                flag = "  <-- SUPERSEDED SINCE THE DECISION"
            sim = f"{c.similarity:+.3f}" if c.similarity is not None else "  n/a "
            print(f"    rank={c.rank} sim={sim} status_now={c.status_now:10} "
                  f"{c.content}{flag}")

        hr("5. rewind_and_rerun -- would it still decide that?")
        faithful = rewind_and_rerun(db, "demo", decision.id, triage_agent)
        print("    (a) replayed at the DECISION timestamp (fidelity check)")
        print(f"        old={faithful.old_action!r}  new={faithful.new_action!r}  "
              f"changed={faithful.action_changed}")
        print("        the past reproduces exactly -- replay is faithful")

        today = rewind_and_rerun(
            db, "demo", decision.id, triage_agent,
            as_of=replay_window_bounds(db).latest,
        )
        print("\n    (b) re-run against TODAY's knowledge")
        print(f"        old={today.old_action!r}  new={today.new_action!r}  "
              f"changed={today.action_changed}")
        print(f"        new rationale: {today.new_rationale}")
        d = today.memory_diff
        print(f"\n    memory availability: then={d.then_count} now={d.now_count} "
              f"common={d.common_count}")
        for ref in d.only_then:
            print(f"      available THEN, gone now: {ref.content}")
        for ref in d.only_now:
            print(f"      learned SINCE:            {ref.content}")

        hr("6. Logical vs physical replay (they answer different questions)")
        with psycopg.connect(dsn) as conn:
            t = conn.execute("SELECT now()").fetchone()[0]
        logical = replay_branch_at(db, "demo", "main", t)
        physical = replay_cluster_at(db, "demo", t)
        print(f"    logical  (what the branch logically contained): {len(logical)} active")
        print(f"    physical (what the cluster physically held):    {len(physical)} rows")
        print("    logical honours retraction; physical shows the stored bytes,")
        print("    including the retracted row that is still physically present.")

        hr("7. GC-window honesty")
        w = replay_window_bounds(db)
        ancient = w.earliest - timedelta(hours=1)

        # Demo staging only (not part of the kernel's behaviour): everything above
        # was created seconds ago, so without this there is nothing for logical
        # replay to *find* an hour past the window and the contrast below lands as
        # "0 memories". Backdate one row so the point is visible. Same technique as
        # tests/test_replay.py::test_logical_replay_works_for_a_branch_older_than_
        # the_gc_window.
        with psycopg.connect(dsn, autocommit=True) as conn:
            branch_id = conn.execute(
                "SELECT id FROM branches WHERE name = 'main'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO memories (branch_id, kind, content, created_at) "
                "VALUES (%s, 'fact', %s, %s)",
                (
                    branch_id,
                    "the alpha cluster was decommissioned",
                    ancient - timedelta(minutes=30),
                ),
            )

        print(f"    gc.ttlseconds = {w.gc_ttl_seconds}")
        print(f"    physical replay safe from {w.earliest:%Y-%m-%d %H:%M:%S} "
              f"to {w.latest:%Y-%m-%d %H:%M:%S}")
        print(f"    (backdated one memory to {ancient - timedelta(minutes=30):%H:%M:%S}, "
              f"outside that window, so the contrast below is visible)")
        try:
            replay_cluster_at(db, "demo", ancient)
        except ReplayWindowExpiredError as exc:
            print("\n    replay_cluster_at() 1h past the window:")
            print(f"      ReplayWindowExpiredError: {str(exc)[:150]}...")
        old_logical = replay_branch_at(db, "demo", "main", ancient)
        print(f"\n    replay_branch_at() at the same instant: OK "
              f"({len(old_logical)} memories) -- logical replay is not GC-bounded")
        for m in old_logical:
            print(f"      recovered: {m.content}")

        hr("8. Audit log")
        with psycopg.connect(dsn) as conn:
            for op, n in conn.execute(
                "SELECT op, count(*) FROM audit_log GROUP BY op ORDER BY op"
            ).fetchall():
                print(f"    {op:18} {n}")
    finally:
        db.close()
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {dbname} CASCADE")


if __name__ == "__main__":
    main()

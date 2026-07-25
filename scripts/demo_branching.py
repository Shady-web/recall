"""End-to-end demo of Recall's branching engine.

Shows: fork -> divergent writes on both sides -> recall returning DIFFERENT
results per branch -> diff -> commit (clean) -> commit (conflict, as data).

Runs against a local CockroachDB with the deterministic fake embedder, so it
needs no AWS credentials:

    docker run -d --name recall-crdb -p 26257:26257 \\
        cockroachdb/cockroach:latest-v25.2 start-single-node --insecure
    python scripts/demo_branching.py
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse, urlunparse

import psycopg

from kernel.branching import commit, diff, discard, fork, get_branch, resolve_ancestry
from kernel.db import Database
from kernel.embeddings import FakeEmbeddingProvider
from kernel.memory import MemoryKernel
from kernel.migrate import migrate

BASE = os.environ.get(
    "RECALL_TEST_DSN", "postgresql://root@localhost:26257/defaultdb?sslmode=disable"
)


def _with_db(dsn: str, name: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{name}"))


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(kernel: MemoryKernel, branch: str, query: str) -> None:
    hits = kernel.recall(branch, query, k=5)
    print(f"  recall({branch!r}, {query!r}) -> {len(hits)} hit(s)")
    for h in hits:
        print(f"    #{h.rank}  sim={h.similarity:+.3f}  {h.memory.content}")


def main() -> None:
    dbname = f"recall_demo_{uuid.uuid4().hex[:8]}"
    admin = _with_db(BASE, "defaultdb")
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
        c.execute(f"CREATE DATABASE {dbname}")
    dsn = _with_db(BASE, dbname)
    migrate(dsn=dsn)

    db = Database(dsn)
    k = MemoryKernel(db, actor="demo", read_only=False, embedder=FakeEmbeddingProvider())

    try:
        hr("1. Baseline knowledge on 'main'")
        base = k.remember("main", "the payment service timeout is 30 seconds", kind="fact")
        k.remember("main", "the payment service runs on port 8080", kind="fact")
        k.remember("main", "checkout latency spiked at 14:00", kind="incident")
        for m in k.list_memories("main"):
            print(f"    {m.kind:9} {m.content}")

        hr("2. Fork 'main' -> 'hotfix' (safe speculation)")
        child = fork(db, "demo", "main", "hotfix")
        print(f"    forked at {child.fork_point_ts}")
        print("    ancestry of 'hotfix':")
        for seg in resolve_ancestry(db, "hotfix"):
            bound = seg.visible_as_of or "live (no bound)"
            print(f"      depth={seg.depth}  {seg.name:8} visible_as_of={bound}")

        hr("3. Divergent writes")
        print("  on 'hotfix': revise the timeout, add a hypothesis")
        k.supersede(base.id, "the payment service timeout is 5 seconds", branch="hotfix")
        k.remember("hotfix", "hypothesis: the timeout causes checkout latency", kind="hypothesis")
        print("  on 'main': an unrelated new fact arrives after the fork")
        k.remember("main", "the payment service added a retry queue", kind="fact")

        hr("4. Recall on both branches -> DIFFERENT results")
        show(k, "main", "payment service timeout")
        print()
        show(k, "hotfix", "payment service timeout")
        print("\n  Note: 'hotfix' sees its own 5s revision (the 30s original is")
        print("  shadowed there) and does NOT see main's post-fork retry queue.")
        print("  'main' still sees 30s -- the branch never touched it.")

        hr("5. diff('main', 'hotfix')")
        d = diff(db, "demo", "main", "hotfix")
        for side in (d.a, d.b):
            print(f"    {side.name:8} added={len(side.added)} "
                  f"superseded={len(side.superseded)} retracted={len(side.retracted)}")

        hr("6. commit('hotfix') -> replay onto 'main'")
        result = commit(db, "demo", "hotfix")
        print(f"    committed={result.committed}  "
              f"replayed={len(result.replayed_memory_ids)}  "
              f"overrides_applied={len(result.applied_override_ids)}  "
              f"conflicts={len(result.conflicts)}")
        print(f"    branch status is now {get_branch(db, 'hotfix').status!r}")
        print()
        show(k, "main", "payment service timeout")

        hr("7. Conflict case: two branches touch the same memory")
        port = [m for m in k.list_memories("main") if "port 8080" in m.content][0]
        fork(db, "demo", "main", "left")
        fork(db, "demo", "main", "right")
        k.supersede(port.id, "the payment service runs on port 9090", branch="left")
        k.supersede(port.id, "the payment service runs on port 7070", branch="right")

        first = commit(db, "demo", "left")
        second = commit(db, "demo", "right")
        print(f"    commit('left')  -> committed={first.committed} "
              f"conflicts={len(first.conflicts)}")
        print(f"    commit('right') -> committed={second.committed} "
              f"conflicts={len(second.conflicts)}")
        for c in second.conflicts:
            print(f"      CONFLICT on {str(c.memory_id)[:8]}: "
                  f"branch={c.branch_status} parent={c.parent_status}")
            print(f"        {c.reason}")
        print(f"    'right' left open (no partial state): "
              f"{get_branch(db, 'right').status!r}")

        hr("8. discard('right') -- nothing is ever hard-deleted")
        discard(db, "demo", "right", reason="superseded by left")
        print(f"    status={get_branch(db, 'right').status!r}")
        with psycopg.connect(dsn) as conn:
            n = conn.execute(
                "SELECT count(*) FROM memories WHERE branch_id = %s",
                (get_branch(db, "right").id,),
            ).fetchone()[0]
            audits = conn.execute(
                "SELECT op, count(*) FROM audit_log GROUP BY op ORDER BY op"
            ).fetchall()
        print(f"    its {n} memory row(s) still on disk, readable for audit/replay")

        hr("9. Audit log (every op, same transaction as the operation)")
        for op, count in audits:
            print(f"    {op:16} {count}")
    finally:
        db.close()
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {dbname} CASCADE")


if __name__ == "__main__":
    main()

"""Recall latency benchmark.

Loads N synthetic memories (default 50,000) into a dedicated database, builds the
vector index, then times many branch-scoped ANN recall queries and reports p50 /
p95 / p99 latency. Uses the deterministic FakeEmbeddingProvider, so it runs with
no AWS credentials or cost and is fully reproducible.

The measured statement is the index-accelerated retrieval query — the same
`WITH candidates AS (... ORDER BY embedding <-> q LIMIT ...)` shape the kernel's
recall() uses — so the numbers reflect the vector index doing real work
(embedding the query text and the per-query audit write are excluded to isolate
retrieval).

Usage:
    python benchmarks/bench_recall.py                    # 50k rows, local cluster
    python benchmarks/bench_recall.py --count 10000 --queries 200
    python benchmarks/bench_recall.py --dsn <dsn> --out benchmarks/recall_benchmark.md
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import statistics
import time
from datetime import UTC, datetime
from urllib.parse import urlparse, urlunparse

import psycopg

from kernel.embeddings import FakeEmbeddingProvider, vector_literal
from kernel.migrate import migrate

_VOCAB = (
    "database server crashed restart deploy rollback latency spike memory leak "
    "disk full cpu throttled network partition timeout retry queue backlog cache "
    "miss replica lag failover incident alert pager oncall dashboard metric trace "
    "log error exception panic degraded recovered healthy nominal saturation"
).split()


def _with_db(dsn: str, dbname: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{dbname}"))


def _synthetic_text(rng: random.Random) -> str:
    return " ".join(rng.choices(_VOCAB, k=rng.randint(6, 14)))


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def run_benchmark(base_dsn: str, count: int, queries: int, k: int) -> dict:
    embedder = FakeEmbeddingProvider()
    rng = random.Random(1234)  # reproducible

    dbname = "recall_bench"
    admin = _with_db(base_dsn, "defaultdb")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
        conn.execute(f"DROP DATABASE IF EXISTS {dbname} CASCADE")
        conn.execute(f"CREATE DATABASE {dbname}")

    dsn = _with_db(base_dsn, dbname)
    migrate(dsn=dsn)

    with psycopg.connect(dsn, autocommit=True) as conn:
        branch_id = conn.execute(
            "SELECT id FROM branches WHERE name = 'main'"
        ).fetchone()[0]

        # Bulk load with the index dropped, then rebuild once — far faster than
        # maintaining the index across N individual inserts.
        print(f"dropping vector index and bulk-loading {count} rows via COPY...")
        conn.execute("DROP INDEX memories@vec_memories_embedding")
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            with cur.copy(
                "COPY memories (branch_id, kind, content, embedding) FROM STDIN"
            ) as copy:
                for _ in range(count):
                    text = _synthetic_text(rng)
                    copy.write_row(
                        (branch_id, "fact", text, vector_literal(embedder.embed(text)))
                    )
        load_s = time.perf_counter() - t0
        print(f"loaded in {load_s:.1f}s; building vector index...")

        t0 = time.perf_counter()
        conn.execute(
            "CREATE VECTOR INDEX vec_memories_embedding ON memories (branch_id, embedding)"
        )
        index_s = time.perf_counter() - t0
        print(f"index built in {index_s:.1f}s; running {queries} recall queries...")

    # Time the retrieval query (index-accelerated ANN + filter), like recall().
    overfetch = max(k * 8, k)
    latencies_ms: list[float] = []
    with psycopg.connect(dsn) as conn:
        # Warm up.
        for _ in range(10):
            _time_one_query(conn, branch_id, embedder, rng, k, overfetch)
        for _ in range(queries):
            latencies_ms.append(
                _time_one_query(conn, branch_id, embedder, rng, k, overfetch)
            )

    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {dbname} CASCADE")

    return {
        "count": count,
        "queries": queries,
        "k": k,
        "load_seconds": round(load_s, 1),
        "index_build_seconds": round(index_s, 1),
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "p99_ms": round(_percentile(latencies_ms, 99), 2),
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


def _time_one_query(conn, branch_id, embedder, rng, k, overfetch) -> float:
    query_literal = vector_literal(embedder.embed(_synthetic_text(rng)))
    sql = (
        "WITH candidates AS ("
        "  SELECT id, status, embedding <-> %s::VECTOR AS _dist "
        "  FROM memories WHERE branch_id = %s "
        "  ORDER BY embedding <-> %s::VECTOR LIMIT %s"
        ") "
        "SELECT id, 1.0 - (POWER(_dist, 2) / 2.0) AS similarity "
        "FROM candidates WHERE _dist IS NOT NULL AND status = 'active' "
        "ORDER BY _dist LIMIT %s"
    )
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (query_literal, branch_id, query_literal, overfetch, k))
        cur.fetchall()
    return (time.perf_counter() - start) * 1000.0


def _write_report(path: str, result: dict) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Recall latency benchmark",
        "",
        f"_Generated {now}_",
        "",
        "Branch-scoped ANN recall against CockroachDB's vector index "
        "(L2, prefix column `branch_id`), using the deterministic "
        "`FakeEmbeddingProvider`. Latency is the retrieval query only "
        "(query embedding and the per-recall audit write are excluded).",
        "",
        "## Environment",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "",
        "## Parameters",
        "",
        f"- Memories loaded: **{result['count']:,}**",
        f"- Recall queries timed: **{result['queries']:,}**",
        f"- k (results per query): **{result['k']}**, "
        f"over-fetch: {max(result['k'] * 8, result['k'])}",
        "",
        "## Results",
        "",
        "| metric | value |",
        "|---|---|",
        f"| p50 | {result['p50_ms']} ms |",
        f"| p95 | {result['p95_ms']} ms |",
        f"| p99 | {result['p99_ms']} ms |",
        f"| mean | {result['mean_ms']} ms |",
        f"| min | {result['min_ms']} ms |",
        f"| max | {result['max_ms']} ms |",
        "",
        f"Bulk load: {result['load_seconds']}s · index build: "
        f"{result['index_build_seconds']}s.",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote report to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall latency benchmark.")
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "RECALL_TEST_DSN",
            "postgresql://root@localhost:26257/defaultdb?sslmode=disable",
        ),
    )
    parser.add_argument("--out", default="benchmarks/recall_benchmark.md")
    args = parser.parse_args()

    result = run_benchmark(args.dsn, args.count, args.queries, args.k)
    print(result)
    _write_report(args.out, result)


if __name__ == "__main__":
    main()

"""Hybrid, branch-scoped retrieval for Recall.

`recall` runs vector similarity search together with structured SQL filters in a
single query — never fetch-then-filter-in-Python. Because CockroachDB's vector
index only accelerates a prefix-scoped ``ORDER BY embedding <-> q LIMIT n`` (and
falls back to a full scan the moment a non-prefix filter appears in the same
scan), we use the standard ANN-plus-filter shape:

    WITH candidates AS (            -- index-accelerated ANN, branch prefix only
        SELECT ..., embedding <-> q AS dist
        FROM memories WHERE branch_id = $b
        ORDER BY embedding <-> q LIMIT $overfetch
    )
    SELECT ... FROM candidates      -- structured filters applied in SQL
    WHERE status = ... AND kind = ...
    ORDER BY dist LIMIT $k

The inner CTE uses the vector index; the outer query applies the structured
filters. We over-fetch candidates so post-filtering still tends to yield ``k``
results. Similarity is cosine, derived from L2 distance on unit vectors:
``1 - ‖a-b‖²/2`` (see migrations/002_vector_index.sql).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from kernel import audit
from kernel.db import Database
from kernel.embeddings import EmbeddingProvider, vector_literal
from kernel.errors import EmbeddingError
from kernel.memory import MEMORY_COLUMNS, resolve_branch
from kernel.models import Memory, RecallResult

# How many ANN candidates to pull per requested result before structured
# filtering. Over-fetching compensates for candidates dropped by the outer
# filters. Tunable; larger = more accurate under selective filters, slower.
_OVERFETCH = 8


def recall(
    db: Database,
    embedder: EmbeddingProvider | None,
    actor: str,
    branch: str,
    query: str,
    *,
    k: int = 10,
    kind: str | None = None,
    min_confidence: float | None = None,
    since: datetime | None = None,
    status: str | None = "active",
) -> list[RecallResult]:
    """Return the ``k`` memories on ``branch`` most similar to ``query``.

    Every call writes one audit row recording the query and the ids returned.
    """
    if embedder is None:
        raise EmbeddingError("recall requires an embedding provider")

    # Embed OUTSIDE the transaction so serialization-failure retries of the DB
    # work do not re-invoke the (paid, network) embedding call.
    query_vec = embedder.embed(query)
    query_literal = vector_literal(query_vec)  # floats only — safe to inline
    overfetch = max(k * _OVERFETCH, k)

    def work(conn) -> list[RecallResult]:
        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)

            # ---- ancestry seam -------------------------------------------
            # TODO(Phase 3): ancestry resolution. Today recall matches exactly
            # one branch_id. In Phase 3 the candidate CTE's WHERE clause becomes
            # the branch's full ancestry — each ancestor branch read AS OF its
            # fork point — e.g. `WHERE branch_id = ANY(<ancestry_ids>)` (branch_id
            # is the vector index prefix, so ANN stays index-accelerated) plus a
            # per-branch AS OF SYSTEM TIME bound. This single line is the seam.
            branch_filter_sql = "branch_id = %s"
            branch_filter_params: list[Any] = [b["id"]]

            # Structured filters applied AFTER the ANN candidate fetch.
            # `_dist IS NOT NULL` drops any null-embedding row (its distance is
            # NULL) here in the outer query rather than in the inner scan, where a
            # non-prefix predicate would defeat the vector index.
            outer_clauses: list[str] = ["_dist IS NOT NULL"]
            outer_params: list[Any] = []
            if status is not None:
                outer_clauses.append("status = %s")
                outer_params.append(status)
            if kind is not None:
                outer_clauses.append("kind = %s")
                outer_params.append(kind)
            if min_confidence is not None:
                outer_clauses.append("confidence >= %s")
                outer_params.append(min_confidence)
            if since is not None:
                outer_clauses.append("created_at >= %s")
                outer_params.append(since)
            outer_where = " AND ".join(outer_clauses) if outer_clauses else "TRUE"

            sql = (
                f"WITH candidates AS ("
                f"  SELECT {MEMORY_COLUMNS}, "
                f"         embedding <-> '{query_literal}'::VECTOR AS _dist "
                f"  FROM memories "
                f"  WHERE {branch_filter_sql} "
                f"  ORDER BY embedding <-> '{query_literal}'::VECTOR "
                f"  LIMIT %s "
                f") "
                f"SELECT {MEMORY_COLUMNS}, 1.0 - (POWER(_dist, 2) / 2.0) AS similarity "
                f"FROM candidates "
                f"WHERE {outer_where} "
                f"ORDER BY _dist "
                f"LIMIT %s"
            )
            params = branch_filter_params + [overfetch] + outer_params + [k]
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = [
                RecallResult(
                    memory=Memory.model_validate(row),
                    similarity=float(row["similarity"]),
                    rank=rank,
                )
                for rank, row in enumerate(rows, start=1)
            ]

            audit.record(
                conn,
                actor=actor,
                op="recall",
                target_type="branch",
                target_id=b["id"],
                payload={
                    "query": query,
                    "k": k,
                    "kind": kind,
                    "min_confidence": min_confidence,
                    "since": since.isoformat() if since is not None else None,
                    "status": status,
                    "returned_ids": [str(r.memory.id) for r in results],
                    "count": len(results),
                },
            )
            return results

    return db.run_in_transaction(work)

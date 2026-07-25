"""Hybrid, ancestry-scoped retrieval for Recall.

`recall` runs vector similarity together with structured SQL filters AND branch
ancestry resolution in a single statement over ``memories`` — never
fetch-then-filter-in-Python.

Query shape::

    WITH candidates AS (          -- index-accelerated ANN over the whole chain
        SELECT ..., embedding <-> q AS _dist
        FROM memories
        WHERE branch_id IN (<resolved ancestry ids>)   -- literal list, keeps
        ORDER BY embedding <-> q                        -- the vector index
        LIMIT overfetch
    )
    SELECT ...
    FROM candidates c
    JOIN (VALUES (branch_id, visible_as_of, depth), ...) AS anc ON ...
    WHERE (anc.visible_as_of IS NULL OR c.created_at <= anc.visible_as_of)
      AND <effective status as of the bound> = ...
      AND <structured filters>
    ORDER BY c._dist LIMIT k

Two properties are load-bearing, both verified empirically against CockroachDB
v25.2 (see the design notes in :mod:`kernel.branching`):

* The ancestry ids must appear as a **literal list**. Writing
  ``branch_id IN (SELECT id FROM ancestry_cte)`` makes the planner abandon the
  vector index and full-scan. So :func:`kernel.branching.resolve_ancestry` runs
  the recursive CTE over the tiny ``branches`` table first, and its result is
  passed in as parameters. The read of ``memories`` itself remains one query.
* Non-prefix predicates in the same scan as the ANN also defeat the index, so
  every bound/filter is applied in the OUTER query over the candidate set.

Effective status is computed **as of each segment's bound**, not read from the
denormalized ``status`` column, so a descendant sees the parent exactly as it was
at the fork point. A branch-local ``memory_overrides`` row (from the nearest
ancestry branch that has one) takes precedence over the row's own timestamps.
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

# ANN candidates fetched per requested result before filtering. Over-fetching
# compensates for candidates dropped by ancestry bounds and structured filters.
_OVERFETCH = 8

# Status of a candidate as of its segment's visibility bound. An override from
# the nearest ancestry branch wins; otherwise fall back to the row's own
# supersede/retract timestamps compared against the bound.
_EFFECTIVE_STATUS = """
COALESCE(
    (SELECT o.status
       FROM memory_overrides o
       JOIN anc a2 ON a2.branch_id = o.branch_id
      WHERE o.memory_id = c.id
        AND (a2.visible_as_of IS NULL OR o.created_at <= a2.visible_as_of)
      ORDER BY a2.depth
      LIMIT 1),
    CASE
        WHEN c.retracted_at IS NOT NULL
             AND (anc.visible_as_of IS NULL OR c.retracted_at <= anc.visible_as_of)
            THEN 'retracted'
        WHEN c.superseded_at IS NOT NULL
             AND (anc.visible_as_of IS NULL OR c.superseded_at <= anc.visible_as_of)
            THEN 'superseded'
        ELSE 'active'
    END
)
"""


def build_recall_sql(
    segments: list, query_literal: str, outer_where: str
) -> str:
    """Assemble the ancestry-scoped recall statement.

    Split out so tests can EXPLAIN the exact SQL the kernel runs and assert the
    vector index is still used.
    """
    branch_placeholders = ", ".join(["%s::UUID"] * len(segments))
    anc_values = ", ".join(["(%s::UUID, %s::TIMESTAMPTZ, %s::INT)"] * len(segments))
    return (
        f"WITH candidates AS ("
        f"  SELECT {MEMORY_COLUMNS}, superseded_at, retracted_at, "
        f"         embedding <-> '{query_literal}'::VECTOR AS _dist "
        f"  FROM memories "
        f"  WHERE branch_id IN ({branch_placeholders}) "
        f"  ORDER BY embedding <-> '{query_literal}'::VECTOR "
        f"  LIMIT %s "
        f"), "
        f"anc (branch_id, visible_as_of, depth) AS (VALUES {anc_values}) "
        f"SELECT c.id, c.branch_id, c.kind, c.content, c.source, c.confidence, "
        f"       {_EFFECTIVE_STATUS} AS status, "
        f"       c.superseded_by, c.metadata, c.created_at, "
        f"       1.0 - (POWER(c._dist, 2) / 2.0) AS similarity "
        f"  FROM candidates c "
        f"  JOIN anc ON anc.branch_id = c.branch_id "
        f" WHERE c._dist IS NOT NULL "
        f"   AND (anc.visible_as_of IS NULL OR c.created_at <= anc.visible_as_of) "
        f"   AND {outer_where} "
        f" ORDER BY c._dist "
        f" LIMIT %s"
    )


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
    """Return the ``k`` memories visible from ``branch`` most similar to ``query``.

    Visibility follows the branch's full ancestry chain (see
    :mod:`kernel.branching`). Every call writes one audit row recording the query
    and the ids returned.
    """
    if embedder is None:
        raise EmbeddingError("recall requires an embedding provider")

    # Embed OUTSIDE the transaction so serialization retries never re-invoke the
    # (paid, network) embedding call.
    query_literal = vector_literal(embedder.embed(query))
    overfetch = max(k * _OVERFETCH, k)

    def work(conn) -> list[RecallResult]:
        from kernel.branching import resolve_ancestry

        with conn.cursor(row_factory=dict_row) as cur:
            b = resolve_branch(cur, branch)
            segments = resolve_ancestry(db, b["id"], conn=conn)

            # Structured filters, applied in the outer query so the inner ANN
            # scan keeps the vector index.
            clauses: list[str] = []
            filter_params: list[Any] = []
            if status is not None:
                clauses.append(f"{_EFFECTIVE_STATUS} = %s")
                filter_params.append(status)
            if kind is not None:
                clauses.append("c.kind = %s")
                filter_params.append(kind)
            if min_confidence is not None:
                clauses.append("c.confidence >= %s")
                filter_params.append(min_confidence)
            if since is not None:
                clauses.append("c.created_at >= %s")
                filter_params.append(since)
            outer_where = " AND ".join(clauses) if clauses else "TRUE"

            sql = build_recall_sql(segments, query_literal, outer_where)
            params: list[Any] = [s.branch_id for s in segments]
            params.append(overfetch)
            for s in segments:
                params.extend([s.branch_id, s.visible_as_of, s.depth])
            params.extend(filter_params)
            params.append(k)

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
                    "ancestry_depth": len(segments),
                    "returned_ids": [str(r.memory.id) for r in results],
                    "count": len(results),
                },
            )
            return results

    return db.run_in_transaction(work)

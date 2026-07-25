"""Backfill embeddings for memories written before Phase 2.

Any memory row with a NULL ``embedding`` (e.g. created during Phase 1) is
embedded and updated in place. Each update writes an audit row in the same
transaction, preserving the kernel's audit invariant. Safe to re-run: it only
touches rows that still lack an embedding.

Usage:
    python -m kernel.backfill                     # real Bedrock provider, config DSN
    python -m kernel.backfill --dsn <dsn> --batch-size 200
"""

from __future__ import annotations

import argparse
import logging

from psycopg.rows import dict_row

from kernel import audit
from kernel.db import Database
from kernel.embeddings import EmbeddingProvider, vector_literal

logger = logging.getLogger("recall.kernel.backfill")


def backfill_embeddings(
    db: Database,
    embedder: EmbeddingProvider,
    *,
    actor: str = "backfill",
    batch_size: int = 100,
) -> int:
    """Embed all memories with a NULL embedding. Returns the number updated."""
    total = 0
    while True:
        # Fetch a batch of un-embedded rows (read-only, its own connection).
        with db.transaction(read_only=True) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT id, content FROM memories "
                    "WHERE embedding IS NULL ORDER BY created_at LIMIT %s",
                    (batch_size,),
                )
                rows = cur.fetchall()

        if not rows:
            break

        for row in rows:
            literal = vector_literal(embedder.embed(row["content"]))

            def work(conn, _id=row["id"], _lit=literal) -> None:
                conn.execute(
                    "UPDATE memories SET embedding = %s::VECTOR WHERE id = %s",
                    (_lit, _id),
                )
                audit.record(
                    conn,
                    actor=actor,
                    op="embed_backfill",
                    target_type="memory",
                    target_id=_id,
                    payload={},
                )

            db.run_in_transaction(work)
            total += 1

        logger.info("backfilled %d memories so far", total)

    logger.info("backfill complete: %d memories embedded", total)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Backfill missing memory embeddings.")
    parser.add_argument("--dsn", default=None, help="CockroachDB DSN (default: config).")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--actor", default="backfill")
    args = parser.parse_args()

    from kernel.config import settings
    from kernel.embeddings import BedrockEmbeddingProvider

    dsn = args.dsn or settings.crdb_connection_string
    db = Database(dsn)
    try:
        count = backfill_embeddings(
            db,
            BedrockEmbeddingProvider(),
            actor=args.actor,
            batch_size=args.batch_size,
        )
    finally:
        db.close()
    print(f"Backfilled {count} memories.")


if __name__ == "__main__":
    main()

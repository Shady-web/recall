"""CLI for the incident-triage agent.

    python -m agent "checkout p99 latency is 4.2s, payments-svc timing out"
    python -m agent --offline --commit "..."

``--offline`` selects the deterministic rule reasoner and the fake embedding
provider, so a run costs nothing and needs no AWS credentials. Without it the
agent calls Bedrock for both embeddings and reasoning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agent.reasoning import ReasoningError, build_reasoner
from agent.triage import IncidentTriageAgent
from kernel.db import Database
from kernel.embeddings import BedrockEmbeddingProvider, FakeEmbeddingProvider
from kernel.errors import RecallError
from kernel.memory import MemoryKernel


def build_kernel(offline: bool, dsn: str | None = None) -> MemoryKernel:
    """Construct a kernel wired for either offline or live operation."""
    from kernel.config import settings

    db = Database(dsn) if dsn else Database()
    embedder = FakeEmbeddingProvider() if offline else BedrockEmbeddingProvider()
    kernel = MemoryKernel(
        db,
        actor=settings.recall_actor_id,
        read_only=settings.recall_read_only,
        embedder=embedder,
    )
    from kernel.db import verify_embedding_dimension, verify_embedding_provider

    verify_embedding_dimension(db, embedder)
    verify_embedding_provider(db, embedder)
    return kernel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    parser.add_argument("incident", help="the incident text to triage")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the fake embedder and the rule reasoner (no AWS calls, no cost)",
    )
    parser.add_argument("--branch", default="main", help="base branch to fork from")
    parser.add_argument("--dsn", default=None, help="override CRDB_CONNECTION_STRING")
    parser.add_argument("-k", type=int, default=6, help="how many memories to recall")
    parser.add_argument(
        "--commit", action="store_true", help="commit the fork back into the base branch"
    )
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        kernel = build_kernel(args.offline, args.dsn)
        agent = IncidentTriageAgent(
            kernel, build_reasoner(offline=args.offline), base_branch=args.branch, k=args.k
        )
        result = agent.run(args.incident, commit=args.commit)
    except (RecallError, ReasoningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.summary(), indent=2))
        return 0

    print()
    print(f"  branch    {result.branch.name}")
    print(f"  reasoner  {result.reasoner}")
    print(f"  recalled  {len(result.recalled)} memories")
    for hit in result.recalled:
        print(f"     {hit.rank}. [{hit.similarity:.3f}] {hit.memory.content[:88]}")
    print(f"  decision  {result.decision.id}")
    print(f"  action    {result.action}")
    if result.rationale:
        print(f"  rationale {result.rationale}")
    if args.commit:
        print(
            f"  commit    committed={result.committed} "
            f"conflicts={result.commit_conflicts}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recall *quality* benchmark — does the right memory come back, and how clearly?

`bench_recall.py` measures how fast retrieval is. This measures whether it is
any good: for a set of incident queries with known-relevant memories, it reports
the rank of each relevant memory and the margin separating relevant from
irrelevant ones.

WHY THIS EXISTS
---------------
A recall system can be fast, correctly indexed, dimensionally consistent, and
still useless — every score can be numerically fine and semantically noise. That
happened in this project: a corpus embedded with the fake provider was queried
with Titan, and recall returned six confident-looking hits scoring 0.040 down to
-0.011. Nothing errored. The only tell was that the numbers "looked low", which
is not a check anyone should have to run by eye.

Migration 004 and `kernel.db.verify_embedding_provider` make that specific
failure impossible now. This benchmark covers the broader question they cannot:
given the *right* embedding space, is the corpus actually separable? That is a
property of the content and the queries, not of the plumbing, and the only
honest way to know it is to measure it.

WHAT IT REPORTS
---------------
* **rank** of each known-relevant memory (1 is best) — the number that decides
  whether the agent sees the runbook it needs inside its top-k.
* **margin** = (worst relevant score) − (best irrelevant score). Positive means
  relevant memories are cleanly separated; near zero means the ordering is
  luck; negative means an irrelevant memory outranks a relevant one.
* **spread** across the whole corpus. A tightly clustered spread means the
  embedding is not discriminating, even when the ranking happens to be right.

Ablations, so the answer is "what moved it" rather than "it works now":
  --dims          512 / 1024 — is the default output width costing separation?
  --queries       compare phrasings of the same incident
  --no-normalize  confirm the unit-vector assumption the L2 index relies on

Usage:
    python benchmarks/recall_quality.py --provider fake        # free, no AWS
    python benchmarks/recall_quality.py --provider bedrock     # real Titan
    python benchmarks/recall_quality.py --provider bedrock --dims 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_incidents import CORPUS, REPLACEMENT_RUNBOOK  # noqa: E402

# --------------------------------------------------------------------------
# The evaluation set: incident queries plus the memories a competent on-call
# engineer would call relevant. Declared up front and independently of any
# result, so this measures recall rather than rationalising whatever came back.
# --------------------------------------------------------------------------

# Note on RB-031: it is the *replacement* for RB-014 and is labelled relevant
# wherever RB-014 is. An earlier version of this file omitted it, which reported
# every query as "INVERTED" because RB-031 consistently outranked memories that
# were labelled relevant. The embedding was right and the label set was wrong —
# recorded here because a benchmark that silently mislabels its ground truth is
# worse than no benchmark, and the correction was to the labels, never to the
# corpus or the queries.
QUERIES: dict[str, dict] = {
    "terse-alert": {
        "text": (
            "payments-svc checkout p99 latency 4.2s and climbing, pgbouncer "
            "cl_waiting rising"
        ),
        "relevant": {"RB-014", "RB-031", "INC-2211", "RB-021", "CFG-002"},
    },
    "symptom-only": {
        "text": "checkout is slow, p99 latency 4.2 seconds",
        "relevant": {"INC-2211", "RB-014", "RB-031"},
    },
    "with-intent": {
        "text": (
            "payments-svc checkout p99 latency has climbed to 4.2s and pgbouncer "
            "cl_waiting is rising. What should we do to restore latency?"
        ),
        "relevant": {"RB-014", "RB-031", "INC-2211", "RB-021", "CFG-002"},
    },
    "keyword-dense": {
        "text": (
            "pgbouncer connection pool exhaustion, cl_waiting rising, payments-svc "
            "checkout p99 latency 4.2s, max_client_conn"
        ),
        "relevant": {"RB-014", "RB-031", "INC-2211", "RB-021", "CFG-002", "CFG-001"},
    },
    "unrelated-control": {
        # A sanity query: if this scores as highly as the real ones, the
        # embedding is not discriminating and every other result is suspect.
        "text": "how do I rotate the TLS certificate on the edge load balancer",
        "relevant": set(),
    },
}


def corpus() -> list[tuple[str, str]]:
    entries = [(e["ref"], e["content"]) for e in CORPUS]
    entries.append((REPLACEMENT_RUNBOOK["ref"], REPLACEMENT_RUNBOOK["content"]))
    return entries


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


def build_provider(name: str, dims: int, normalize: bool):
    from kernel.embeddings import BedrockEmbeddingProvider, FakeEmbeddingProvider

    if name == "fake":
        if not normalize:
            raise SystemExit("--no-normalize is only meaningful for the bedrock provider")
        return FakeEmbeddingProvider(dimensions=dims)
    return BedrockEmbeddingProvider(dimensions=dims, normalize=normalize)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(provider, entries, queries, k: int) -> dict:
    vectors = {}
    for ref, text in entries:
        vectors[ref] = provider.embed(text)

    norms = [norm(v) for v in vectors.values()]
    report = {
        "space_id": getattr(provider, "space_id", "unknown"),
        "dimensions": provider.dimensions,
        "unit_norm_min": min(norms),
        "unit_norm_max": max(norms),
        "queries": {},
    }

    for name, spec in queries.items():
        qv = provider.embed(spec["text"])
        scored = sorted(
            ((ref, cosine(qv, v)) for ref, v in vectors.items()), key=lambda t: -t[1]
        )
        ranks = {ref: i for i, (ref, _) in enumerate(scored, 1)}
        scores = dict(scored)

        relevant = spec["relevant"]
        irrelevant = set(vectors) - relevant

        rel_scores = [scores[r] for r in relevant] if relevant else []
        irr_scores = [scores[r] for r in irrelevant] if irrelevant else []
        margin = (min(rel_scores) - max(irr_scores)) if rel_scores and irr_scores else None

        report["queries"][name] = {
            "text": spec["text"],
            "top_k": [(ref, round(s, 4)) for ref, s in scored[:k]],
            "relevant_ranks": {r: ranks[r] for r in sorted(relevant)},
            "relevant_in_top_k": sum(1 for r in relevant if ranks[r] <= k),
            "relevant_total": len(relevant),
            "margin": None if margin is None else round(margin, 4),
            "score_spread": round(scored[0][1] - scored[-1][1], 4),
            "best": round(scored[0][1], 4),
            "worst": round(scored[-1][1], 4),
        }
    return report


def render(report: dict, k: int) -> None:
    print(f"\nspace      : {report['space_id']}")
    print(f"dimensions : {report['dimensions']}")
    # The kernel ranks by L2 and derives cosine as 1 - d²/2. That identity holds
    # only for unit vectors, so this line is the check on the assumption the
    # vector index rests on.
    is_unit = abs(report["unit_norm_max"] - 1) < 1e-3
    unit_note = (
        "(unit — L2 ranking == cosine ranking)"
        if is_unit
        else "(NOT unit — 1 - d²/2 is no longer cosine; ranking assumption broken)"
    )
    print(
        f"unit norms : {report['unit_norm_min']:.6f} .. "
        f"{report['unit_norm_max']:.6f}  {unit_note}"
    )

    for name, r in report["queries"].items():
        print(f"\n── {name} ──")
        print(f'   "{r["text"][:78]}"')
        for i, (ref, score) in enumerate(r["top_k"], 1):
            mark = "*" if ref in r["relevant_ranks"] else " "
            print(f"     {i:2}.{mark} {score:+.4f}  {ref}")
        if r["relevant_total"]:
            print(
                f"   relevant in top-{k}: {r['relevant_in_top_k']}/{r['relevant_total']}"
                f"   ranks={r['relevant_ranks']}"
            )
            verdict = (
                "clean separation"
                if (r["margin"] or 0) > 0.05
                else "marginal — ordering is close to luck"
                if (r["margin"] or 0) > 0
                else "INVERTED — an irrelevant memory outranks a relevant one"
            )
            print(f"   margin: {r['margin']:+.4f}  ({verdict})")
        print(f"   spread: {r['score_spread']:.4f}  [{r['worst']:+.4f} .. {r['best']:+.4f}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["fake", "bedrock"], default="fake")
    p.add_argument("--dims", type=int, default=1024)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("-k", type=int, default=6, help="top-k the agent actually sees")
    p.add_argument("--json", type=Path, default=None, help="also write raw results here")
    args = p.parse_args()

    provider = build_provider(args.provider, args.dims, not args.no_normalize)
    report = evaluate(provider, corpus(), QUERIES, args.k)
    render(report, args.k)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Recall latency benchmark

_Generated 2026-07-25 14:09:25Z_

Branch-scoped ANN recall against CockroachDB's vector index (L2, prefix column `branch_id`), using the deterministic `FakeEmbeddingProvider`. Latency is the retrieval query only (query embedding and the per-recall audit write are excluded).

## Environment

- Python: 3.12.3
- Platform: Linux-6.18.5-x86_64-with-glibc2.39

## Parameters

- Memories loaded: **50,000**
- Recall queries timed: **300**
- k (results per query): **10**, over-fetch: 80

## Results

| metric | value |
|---|---|
| p50 | 21.82 ms |
| p95 | 28.61 ms |
| p99 | 42.39 ms |
| mean | 22.41 ms |
| min | 14.66 ms |
| max | 49.85 ms |

Bulk load: 20.9s · index build: 249.8s.

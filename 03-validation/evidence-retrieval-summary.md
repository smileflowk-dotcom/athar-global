# Evidence Retrieval V0 — Runtime Result

## RESULT

FAIL

## Models

- Embed: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Rerank: `nvidia/llama-nemotron-rerank-vl-1b-v2`
- Runtime: NVIDIA hosted NIM endpoints

## Corpus

- Pages extracted: 62
- Chunks: 106
- Chunk chars: 1800
- Overlap chars: 250

## Metrics

- Recall@5 embeddings: 0.800
- Recall@3 after rerank: 0.800
- Top-1 after rerank: 0.600
- Locator preservation: 1.000
- Fabricated evidence passages: 0

## Gate

- Recall@5 >= 0.90
- Recall@3 >= 0.90
- Top-1 >= 0.80
- Locator preservation = 1.00
- Fabricated evidence = 0

## Runtime metadata

- Timestamp UTC: 2026-10-04T09:00:46.255052+00:00
- Corpus embedding latency: 8.16s
- Query embedding latency total: 5.93s
- Rerank latency total: 6.04s

## NEXT

Inspect misses, make one documented correction without changing the gold set, then rerun.

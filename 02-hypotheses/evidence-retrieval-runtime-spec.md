# Runtime Spec — Evidence Retrieval V0

Status: READY FOR IMPLEMENTATION

## Goal

Implement the smallest executable benchmark harness for the already-fixed 10-case gold set.

Pipeline:

`source PDF → page-preserving chunks → embeddings → top 5 → rerank → top 3 / top 1 → metrics`

No UI. No reasoning model. No legal conclusions.

## Runtime

Use Nebius Token Factory.

Official endpoints:
- GET /v1/models
- POST /v1/embeddings
- POST /v1/rerank

Docs:
- https://api.tokenfactory.nebius.com/docs

## NVIDIA model candidates

Preferred text embedding candidate:
- NVIDIA `nemotron-3-embed-1b`

Official model catalog:
- https://build.nvidia.com/models?q=embed

Preferred rerank candidate:
- NVIDIA `llama-nemotron-rerank-vl-1b-v2`

Official model card:
- https://build.nvidia.com/nvidia/llama-nemotron-rerank-vl-1b-v2/modelcard

Important:
Do not hard-code an unverified Nebius model identifier.
At runtime, query `GET /v1/models`, resolve an available NVIDIA embedding model and NVIDIA reranker that match these model families, and record the exact IDs used.

If the required NVIDIA model families are unavailable in Token Factory, stop with a clear BLOCKED result rather than silently substituting another publisher.

## Inputs

- `03-validation/evidence-retrieval-gold.jsonl`
- source PDF URL stored in the gold set

## Chunking

V0 must:
- preserve page number;
- preserve document/source URL;
- create stable chunk IDs;
- never paraphrase source text before retrieval.

Keep chunking simple and deterministic.

## Outputs

Create:

- `03-validation/evidence-retrieval-results.jsonl`
- `03-validation/evidence-retrieval-summary.md`
- raw run metadata with:
  - exact model IDs
  - timestamp
  - chunking parameters
  - latency if available
  - errors

## Metrics

- Recall@5 embeddings
- Recall@3 after rerank
- Top-1 after rerank
- locator preservation rate
- fabricated passage count

PASS gate:
- Recall@5 >= 0.90
- Recall@3 >= 0.90
- Top-1 >= 0.80
- locator preservation = 100%
- fabricated passage count = 0

## Environment

Expected secret:
- `NEBIUS_API_KEY`

Never commit secrets.

## Failure policy

Return BLOCKED if:
- `NEBIUS_API_KEY` is missing;
- NVIDIA embedding/rerank model families cannot be resolved in Token Factory;
- the source PDF cannot be retrieved.

Return FAIL if the benchmark executes but misses the quantitative gate.

Return PASS only if every gate is met.

## Next

Only after PASS:
integrate the retrieval brick into ATHAR architecture and move to document-intelligence validation.

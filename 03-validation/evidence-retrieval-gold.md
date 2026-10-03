# Evidence Retrieval Gold Set V0

Status: MATERIALIZED / READY FOR RUNTIME TEST

## Purpose

Provide a fixed 10-case benchmark before any NVIDIA retrieval runtime is implemented.

Source document:
- European Commission / EuroHPC JU
- MeluXina-AI Tender Specifications — Part 1 Administrative Specifications
- Public tender document, 2026

## Why one document first

V0 isolates retrieval quality from cross-document routing complexity.

If Embed + Rerank cannot retrieve exact evidence reliably inside one real procurement dossier, ATHAR should not add reasoning or UI on top.

## Dataset

`evidence-retrieval-gold.jsonl`

Each case contains:
- id
- query
- source_url
- document
- page
- section
- short exact gold quote
- semantic expectation

The short quote is only an anchor. The runtime benchmark must retrieve the source passage from the actual document/chunk corpus, not from this gold file.

## PASS GATE

- Recall@5 embedding >= 0.90
- Recall@3 after rerank >= 0.90
- Top-1 after rerank >= 0.80
- locator preservation = 100%
- fabricated evidence = 0

## Validation rule

Do not tune prompts or chunks against hidden knowledge of the gold answer after seeing failures without recording the change as a new benchmark version.

## NEXT

Implement the smallest runtime harness that:

1. downloads/loads the source document;
2. chunks it while preserving page locators;
3. calls Nemotron embeddings through Nebius;
4. retrieves top 5;
5. reranks top 5;
6. writes raw results;
7. computes benchmark metrics.

No UI.
No reasoning model.
No final procurement finding generation.

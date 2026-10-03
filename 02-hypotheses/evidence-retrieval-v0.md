# HYPOTHESIS — Evidence Retrieval Brick

## HYPOTHESIS

A two-stage NVIDIA retrieval pipeline can reliably surface the exact procurement evidence needed for a finding:

`query → Nemotron embedding retrieval → Nemotron rerank → top evidence passage`

## RATIONALE

NVIDIA currently exposes:
- `nemotron-3-embed-1b` for semantic retrieval / RAG.
- `llama-nemotron-rerank-vl-1b-v2` for relevance scoring / reranking.
- Nebius Token Factory exposes OpenAI-compatible `/v1/embeddings` and `/v1/rerank` endpoints.

This makes the evidence-retrieval brick independently testable before adding reasoning, UI, or country-specific procurement logic.

## MINIMAL CONTRACT

Input:
- one procurement dossier already split into passages/pages
- one evidence query

Output:
- top 5 embedding candidates
- top 3 reranked candidates
- top 1 selected evidence passage
- page / source locator
- retrieval scores where available

## RISK

- Procurement wording may be highly legalistic or multilingual.
- Relevant evidence may be in tables or scans rather than text passages.
- Embedding-only retrieval may miss exact procedural language.
- Reranking may improve ranking but still return a plausible wrong passage.

## TEST

Use a small gold dataset with known answers.

For each case:
1. provide a procurement evidence query;
2. retrieve top 5 candidates with Nemotron embeddings;
3. rerank those candidates;
4. check whether the gold passage is:
   - in embedding top 5;
   - in reranked top 3;
   - ranked #1 after reranking.

## PASS GATE

Initial MVP gate:
- Recall@5 >= 0.90
- Recall@3 after rerank >= 0.90
- Top-1 accuracy after rerank >= 0.80
- zero fabricated evidence passages
- every result preserves its source/page locator

If the gold dataset is smaller than 10 cases, result is INCONCLUSIVE rather than PASS.

## NOT IN SCOPE

- UI
- final finding generation
- legal conclusions
- fraud detection
- autonomous decisions
- country-specific rule engine
- document OCR / parsing benchmark

## NEXT

Create a 10-case public procurement gold set with exact source passages and page locators.

# Skill — Evidence Retrieval

## WHEN TO USE

Use when ATHAR must find the exact source passage that supports or contradicts a procurement finding.

## INPUT

- evidence query
- passages with stable source/page locators
- optional language / document metadata

## NVIDIA / NEBIUS

Primary candidates:
- Nemotron embedding model
- Nemotron reranking model
- Nebius Token Factory /v1/embeddings
- Nebius Token Factory /v1/rerank

## OUTPUT

- top embedding candidates
- reranked candidates
- selected top evidence passage
- preserved source/page locator
- scores where available

## TEST

Required metrics:
- Recall@5
- Recall@3 after rerank
- Top-1 accuracy
- locator preservation
- fabricated passage count

## FAIL CONDITIONS

FAIL if:
- retrieved text is fabricated;
- locator is lost;
- gold passage recall is below the gate;
- reranking adds no measurable value and still remains in architecture.

## RULE

Return source text; never rewrite evidence into a fake quotation.

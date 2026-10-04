# TEST — Evidence Retrieval V0

## TEST

Validate whether NVIDIA embedding + reranking can retrieve exact evidence from procurement documents.

## EXPECTED

For at least 10 gold-standard queries:

- gold passage appears in top 5 embedding results in >= 90% of cases;
- gold passage appears in top 3 reranked results in >= 90% of cases;
- gold passage is ranked #1 after reranking in >= 80% of cases;
- all returned passages retain exact source/page locators;
- no result invents text that is absent from the source dossier.

## ACTUAL

Not run yet.

## RESULT

INCONCLUSIVE

Reason:
No gold dataset and no live Nebius/NVIDIA retrieval run have been executed yet.

## REQUIRED ARTIFACTS

- `evidence-retrieval-gold.jsonl`
- `evidence-retrieval-results.jsonl`
- `evidence-retrieval-summary.md`

## METRICS

- Recall@5 embedding
- Recall@3 reranked
- Top-1 accuracy
- locator preservation rate
- fabricated-passage count
- optional latency per query
- optional cost per query

## NEXT

Build the 10-case public procurement gold dataset before implementing the runtime call.


## Evaluation correction — PDF spacing artifacts

During the first NVIDIA-hosted run, two retrieved gold passages were on the correct page and contained the correct source text, but PyPDF had inserted spaces inside words (for example `partic ipate` and `subcontr actors`). The evaluator therefore produced false negatives despite successful retrieval.

Correction applied:
- gold answers unchanged;
- page constraint unchanged;
- retrieval candidates unchanged;
- evaluator now ignores whitespace/punctuation differences when checking the verbatim gold anchor.

This is an evaluator robustness fix, not benchmark tuning.

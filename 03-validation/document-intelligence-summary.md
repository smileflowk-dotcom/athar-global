# Document Intelligence V0 — Validation Result

## TEST

Run the simplest page-preserving extraction against the ten existing EuroHPC gold cases.

## EXPECTED

Exact source text anchors, page locators, useful section labels, and zero invented evidence.

## ACTUAL

- Method: `pypdf` direct extraction; no NVIDIA Parse/OCR selected.
- Source pages sampled: 62
- Quote-anchor fidelity: 1.00
- Section-label usefulness: 0.80
- Locator preservation: 1.00
- Fabricated evidence: 0

## RESULT: PASS

The NVIDIA document-intelligence capability remains unselected for this born-digital demo case: direct extraction preserved all tested text and locators, while section labels remained useful on at least 80% of cases. Benchmark Parse/OCR later for scanned or degraded documents.

## NEXT

Use this extraction baseline for the minimal reasoning and end-to-end proof; benchmark Parse/OCR only for scanned or degraded documents.

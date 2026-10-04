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

## RESULT: FAIL

The NVIDIA document-intelligence capability remains unselected for this born-digital demo case unless a later scan/table benchmark shows measurable value.

## NEXT

Use this extraction baseline for the minimal reasoning and end-to-end proof; benchmark Parse/OCR only for scanned or degraded documents.

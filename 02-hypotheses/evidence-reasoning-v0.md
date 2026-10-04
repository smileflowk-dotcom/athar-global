# HYPOTHESIS — Evidence Reasoning V0

## HYPOTHESIS

A constrained NVIDIA Nemotron chat model on Nebius Token Factory can classify a procurement statement as `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE` using only supplied evidence and preserve the document/page locator.

## RATIONALE

Retrieval already supplies exact source passages. A small, structured reasoning contract can test grounded classification without adding legal conclusions, fraud claims, or autonomous decisions.

## RISK

The available Nemotron chat tier may vary by account, and a model may add unsupported language or alter the evidence quotation.

## TEST

Run five fixed cases from the existing EuroHPC evidence, require exact evidence copying and a document/page source, and accept only at least four correct classifications with zero fabricated evidence and zero unsupported legal conclusions.

# Evidence Reasoning V0 — Validation Result

## TEST

Classify five fixed requirement/evidence pairs with a NVIDIA Nemotron chat model through Nebius Token Factory.

## EXPECTED

At least four of five statuses correct, all five document/page locators preserved, zero fabricated evidence, and zero unsupported legal conclusions.

## ACTUAL

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`
- Endpoint: Nebius Token Factory `/v1/chat/completions`
- Status accuracy: 0.00 (0/5)
- Locator preservation: 0/5
- Fabricated evidence: 5
- Unsupported legal conclusions: 5

## RESULT: FAIL

## NEXT

Do not proceed to end-to-end integration; inspect the constrained output failures.

# Operating Guide

## Operating loop

READ → PLAN → ACT → VERIFY → LEARN → COMMIT → NEXT

## ICM routing

- `01-intelligence/` — facts, official documentation, model capabilities, hackathon rules, user needs, and market evidence.
- `02-hypotheses/` — architecture hypotheses, product hypotheses, model choices, workflow designs, and business assumptions.
- `03-validation/` — tests, benchmarks, experiments, PASS/FAIL evidence, and measurements.

## Rules

- Separate facts from assumptions.
- Give every important decision a reason.
- Take one implementation step at a time.
- Validate before expanding.
- Do not build speculative features.
- Do not hide failures.
- Prefer official NVIDIA, Nebius, and hackathon sources.
- Keep the user-facing product simple even if the backend is sophisticated.


## Tool and skill routing

Before implementing a NVIDIA/Nebius capability, read the matching repository skill:

- evidence retrieval → `skills/evidence-retrieval.md`
- document parsing/OCR → `skills/document-intelligence.md`
- reasoning over evidence → `skills/evidence-reasoning.md`
- any benchmark / promotion decision → `skills/benchmark.md`

Authoritative tool maps:
- `tools/nvidia/README.md`
- `tools/nebius/README.md`

Do not select or install a model merely because it exists. Every component must earn integration through validation.

# Skill — Benchmark

## WHEN TO USE

Use before promoting any NVIDIA/Nebius component into ATHAR architecture.

## REQUIRED RECORD

TEST
EXPECTED
ACTUAL
RESULT: PASS / FAIL / INCONCLUSIVE
NEXT

## RULES

- define the gold set before running the model;
- keep raw outputs;
- preserve source locators;
- record model and endpoint used;
- record failures, not only successes;
- fewer than the required sample size = INCONCLUSIVE;
- never promote a component based on demo quality alone.

## CORE BENCHMARK ORDER

1. evidence retrieval
2. document intelligence
3. evidence reasoning
4. end-to-end vertical slice
5. observability / optimization

## DECISION

PASS → integrate
FAIL → fix once or reject
INCONCLUSIVE → gather enough evidence

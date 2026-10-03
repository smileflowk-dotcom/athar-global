# Skill — Evidence Reasoning

## WHEN TO USE

Use only after evidence retrieval has produced source-grounded candidate passages.

## INPUT

- rule / review question
- observed fact
- exact retrieved evidence
- source locators

## NVIDIA / NEBIUS

- Nemotron reasoning model via Nebius Token Factory

Exact model tier remains an experiment.

## OUTPUT

Constrained structured result:
- SUPPORTED
- CONTRADICTED
- INSUFFICIENT_EVIDENCE
- short explanation
- evidence locator references

## SAFETY RULE

The model must not autonomously decide:
- fraud
- illegality
- guilt
- final compliance status

Human validation remains mandatory.

## TEST

Measure:
- groundedness
- unsupported claim rate
- correct insufficient-evidence behavior
- citation / locator fidelity

## FAIL CONDITIONS

Any invented evidence or unsupported legal conclusion is a FAIL.

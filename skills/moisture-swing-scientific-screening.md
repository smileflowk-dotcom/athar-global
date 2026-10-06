# Skill — Moisture-Swing Scientific Screening

## WHEN TO USE

Use for any CO2 moisture-swing candidate generation, ranking, promotion, or validation decision.

## INPUT

- experimental or literature-derived material/property data
- candidate definitions
- operating conditions
- uncertainty estimates
- optional atomistic results

## REQUIRED ORDER

DATA → VALIDATE MODEL → ESTIMATE UNCERTAINTY → ACQUIRE → FEASIBILITY → ATOMISTIC CHECK → INDEPENDENT CROSS-CHECK → LAB

Do not reorder this sequence without a documented reason.

## DECISION RULES

- Never rank by one descriptor alone.
- Prefer moisture-swing working capacity over single-condition uptake.
- Treat architecture, functional group, pore structure, counter-ion, humidity, and CO2 conditions as interacting variables.
- Require grouped validation.
- Require a generalization test such as leave-one-resin-out where possible.
- Record uncertainty and out-of-distribution risk.
- Use active learning to choose the next informative candidates or experiments.
- Keep at least one known control/reference in every batch.

## ACTIVE LEARNING

Compare at least:
- best predicted performance
- highest uncertainty
- balanced performance + uncertainty
- diversity-aware batch selection

A recommendation must state why it was selected.

## ATOMISTIC SCREENING

Only use after the candidate passes the data/feasibility layer.

Required:
- multiple geometries
- convergence gate
- explicit hydration proxy
- no free-energy claim unless free energy is actually computed
- independent method before lab promotion

## OUTPUT

For each candidate:
- predicted performance
- uncertainty
- novelty/OOD flag
- feasibility status
- atomistic status if run
- evidence sources
- promotion gate reached
- next experiment that would reduce uncertainty most

## FAIL CONDITIONS

FAIL if:
- a candidate is called validated without lab data
- ranking depends on one static delta-E
- train/test leakage exists
- uncertainty is ignored
- candidate chemistry is infeasible or undefined
- exact real material structure is unknown but atomistic output is presented as evidence for that material

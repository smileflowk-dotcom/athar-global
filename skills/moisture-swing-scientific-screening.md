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

READ PRIOR STATE → DATA/EVIDENCE → VALIDATE MODEL → ESTIMATE UNCERTAINTY → ACQUIRE → FEASIBILITY → ATOMISTIC CHECK → INDEPENDENT CROSS-CHECK → LAB

Do not reorder this sequence without a documented reason.

## CONTINUITY RULE

Never restart a broad discovery search simply because a new tool or protocol was added.

For each proposed step, state:
- what is already known;
- what remains uncertain;
- why this step reduces that uncertainty;
- what decision will change depending on the result.

If those four points cannot be stated, do not run the step.

## DECISION RULES

- Never rank by one descriptor alone.
- Prefer moisture-swing working capacity over single-condition uptake.
- Treat architecture, functional group, pore structure, counter-ion, humidity, and CO2 conditions as interacting variables.
- Require grouped validation.
- Require a generalization test such as leave-one-resin-out where possible.
- Record uncertainty and out-of-distribution risk.
- Use active learning to choose the next informative candidates or experiments.
- Keep at least one known control/reference in every batch.
- For the current project, treat phosphate-derived chemistries as the priority family and carbonate as the control unless new evidence justifies changing that priority.
- Do not interpret "priority" as "validated winner"; preserve justified non-phosphate comparators.

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

## CODE EXECUTION ENFORCEMENT (V4)

The canonical orchestration is `.github/workflows/chemistry-discovery-v4.yml` with domain-specific rules in `chemistry/v4/scientific_gates.py`.

- Scientific Gate A (experimental Qe grouped holdout and naive baseline) must PASS before physics.
- Gate B must record leave-one-resin-out group-bootstrapped performance and uncertainty.
- Gate C must record exploitation, exploration, balanced and diversity/reference acquisition reasons. Keep carbonate control.
- Gate D must show evidence references, molecular formal charge, neutral site stoichiometry and verified IRA900 Type-I representation. Absent chemical or literature evidence -> HOLD, not a guess.
- Only A–D-passing finalists may receive new NVIDIA ALCHEMI or independent GFN1 calculations; never spend atomistic budget on all virtual configurations.
- 0/9 H2O-only GFN1 results count as PRECHECK, never full robustness. Strong promotion requires target + carbonate convergence in 0/3/6/9 H2O with multiple geometries and matched protocol.
- `PASS_TO_PHYSICS` is **not** a laboratory material approval. No model identifies lab-validated materials without physical adsorption tests.
- Millions of *operating configurations* never imply millions of distinct chemical compounds.
- Previous evidence is conserved even when a prior hypothesis is rejected; never silently run repeats of known negative tests.

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

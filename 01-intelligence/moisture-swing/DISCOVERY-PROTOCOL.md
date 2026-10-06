# Moisture-Swing Discovery Protocol

Status: proving-ground protocol. This does not turn ATHAR Global into a dedicated CO2 materials repo.

## Objective

Identify 3-5 moisture-responsive CO2 sorbent candidates credible enough to justify real laboratory testing.

A candidate is never called a discovery before experimental validation.

## Discovery funnel

1. Public evidence and experimental dataset
2. Baseline ML with grouped validation
3. Leave-one-resin-out generalization test
4. Active-learning acquisition
5. Feasibility / novelty filter
6. Atomistic screening only on shortlisted candidates
7. Independent computational cross-check
8. Top 3-5 lab candidates
9. Experimental validation and feedback into the model

## Search space

A condition is not defined by anion alone. The search space must include, where data exists:

- resin / backbone
- ammonium or functional-group type
- pore architecture
- BET surface area
- pore volume
- nitrogen content
- ion-exchange capacity
- water retention / water uptake
- counter-ion identity
- counter-ion charge and pKa
- temperature
- humidity
- CO2 concentration
- mass/volume or equivalent operating condition

## Primary objectives

Do not optimize a single atomistic energy.

Primary experimental objective:
- moisture-swing working capacity = Qe(low humidity) - Qe(high humidity), under comparable conditions

Secondary objectives:
- absolute CO2 capacity
- adsorption/desorption kinetics
- water uptake
- cyclic stability
- manufacturability
- material and regeneration cost
- process compatibility

## ML protocol

Required before promoting candidates:

- group train/test splits by experimental condition ID
- grouped inner cross-validation
- naive baselines
- multiple random seeds
- leave-one-resin-out test when resin identity is available
- report uncertainty, not only point prediction
- inspect learning curve for data limitation / overfitting

Preferred baseline:
- reuse and reproduce only the parts of the public MSA-ML workflow that reduce uncertainty for the current hypothesis
- do not restart discovery from zero when prior evidence already identifies a high-value chemical family
- compare LightGBM, XGBoost, Random Forest, CatBoost, MLP when the dataset supports it
- use SHAP only after predictive validity is established

## Active-learning protocol

The acquisition layer must balance exploitation and exploration.

For every proposed experiment/candidate record:
- predicted moisture-swing performance
- predictive uncertainty
- novelty / distance from observed data
- experimental feasibility
- synthesis / procurement difficulty
- expected information gain

Do not select solely by maximum predicted Qe or maximum uncertainty.

Initial acquisition modes to compare:
1. exploitation: highest predicted swing
2. exploration: highest uncertainty
3. balanced acquisition: high swing + high uncertainty + feasible
4. diversity-aware batch: avoid selecting near-duplicates in the same round

A batch should contain both likely winners and information-rich probes.

## Atomistic-computation role

ALCHEMI/AIMNet2 is a second-stage filter, not the primary discovery engine.

Use it only after a candidate has survived data-driven and feasibility filters.

Atomistic requirements:
- several initial geometries
- explicit water states
- convergence filtering
- near/far or interaction proxy interpreted only as a proxy
- enough valid pairs before ranking
- independent-method cross-check before laboratory prioritization

Never equate:
- static delta-E with adsorption free energy
- microhydration count with relative humidity
- convergence with physical validity
- NVIDIA reasoning with physical evidence
- a computational winner with a discovered material

## Promotion gates

### Gate A — Data credibility
PASS only if the model beats naive baselines and grouped validation is stable.

### Gate B — Generalization
PASS only if held-out resin / materially different conditions are predicted with useful accuracy or calibrated uncertainty.

### Gate C — Acquisition
PASS only if the candidate is selected by a documented acquisition rule and is not merely an interpolation duplicate.

### Gate D — Feasibility
PASS only if chemistry, synthesis/procurement, safety, and measurement are realistic.

### Gate E — Atomistic support
PASS only if the signal is converged across multiple geometries and survives an independent computational check.

### Gate F — Lab candidate
A material can be called a lab candidate only after A-E pass.

### Gate G — Validated material
Only laboratory data can validate the material.

## Continuity / no-reset rule

The protocol validates and narrows accumulated evidence; it does not erase it.

Before any new campaign:
1. read the current project state and prior validated literature findings;
2. identify exactly which uncertainty the next step reduces;
3. do not rerun a broad screen if the answer is already established;
4. do not expand the search space unless the current priority family has been tested against a justified comparator.

## Current priority hypothesis

The phosphate family is a priority evidence-backed region of the search space, not a proven universal winner.

Priority variants already identified:
- HPO4^2-
- PO4^3-
- P2O7^4-

The key open question is not whether phosphate can work in moisture-swing capture. The open question is which combinations of resin/backbone, ammonium or functional-group type, pore architecture, and phosphate-derived counter-ion are underexplored, feasible, and superior under comparable conditions.

Carbonate remains the control/reference and selected non-phosphate families may be retained only as justified comparators.

## Current status

- V1-V6 are method/infrastructure experiments and must not be treated as a completed discovery campaign.
- Carbonate remains a control/reference.
- No new candidate is currently validated.
- The next step is targeted: build the phosphate-centered candidate space from existing evidence and use ML/uncertainty only to rank or challenge those combinations, not to restart discovery from zero.

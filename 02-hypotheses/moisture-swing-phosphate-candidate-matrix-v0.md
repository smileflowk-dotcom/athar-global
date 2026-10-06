# Phosphate-Centered Candidate Matrix V0

Status: hypothesis matrix for targeted discovery. Not a list of validated candidates.

## Why this matrix exists

The literature already supports phosphate-derived counter-ions as a productive moisture-swing family. The remaining discovery question is which material architecture × functional chemistry × phosphate-derived ion combinations are most promising, underexplored, feasible, and worth expensive validation.

This matrix therefore starts from accumulated evidence rather than restarting a broad ion search.

## Evidence anchors

1. Dibasic phosphate showed the largest mass-normalized moisture-swing capacity in a broad ion comparison.
2. Tribasic phosphate showed the best performance when kinetics were included.
3. Pyrophosphate showed the highest swing capacity when normalized per ion / unit charge.
4. A prior IER-PO4 system reported substantially higher swing capacity than a carbonate analogue.
5. A 2026 structure-property study found a strong interaction between ammonium type and anion:
   - Type I IERs favored phosphate relative to carbonate.
   - Type II IERs favored carbonate relative to phosphate.
6. Macroporous resins can outperform gel analogues when pores are intermediate-sized and well connected.
7. Acrylic backbones tend to take up more water than styrenic ones, which may help or hurt depending on the working-capacity / kinetics tradeoff.

Primary references:
- https://doi.org/10.1021/acs.est.3c02543
- https://doi.org/10.1021/acs.energyfuels.9b00863
- https://doi.org/10.1021/acs.est.5c16820
- https://doi.org/10.1021/acs.est.5c11862

## Search dimensions

Core material dimensions:
- backbone: styrenic / acrylic
- ammonium or strong-base functionality: Type I / Type II
- pore architecture: gel / macroporous
- pore quality: intermediate and connected preferred for macroporous branch
- phosphate-derived counter-ion:
  - HPO4^2- (dibasic phosphate)
  - PO4^3- (tribasic phosphate)
  - P2O7^4- (pyrophosphate)

Controls:
- CO3^2- on the same resin architecture wherever possible

Operating dimensions:
- low vs high RH under matched temperature and CO2 concentration
- ambient-relevant CO2 concentration where experimental data supports it
- working capacity / moisture swing, not single-condition uptake alone

## Candidate families

| ID | Backbone | Functionality | Pore | Ion | Why it is interesting | Current status |
|---|---|---|---|---|---|---|
| P1 | styrenic | Type I | macroporous, intermediate/connected | HPO4^2- | Combines 2026 Type-I/phosphate rule with dibasic phosphate mass-capacity signal | PRIORITY / exact novelty to verify |
| P2 | styrenic | Type I | macroporous, intermediate/connected | PO4^3- | Combines Type-I/phosphate rule with strongest kinetics-adjusted phosphate signal | PRIORITY / exact novelty to verify |
| P3 | styrenic | Type I | macroporous, intermediate/connected | P2O7^4- | Tests high charge-normalized swing in architecture favored by 2026 pore rules | PRIORITY / exact novelty to verify |
| P4 | acrylic | Type I | macroporous, intermediate/connected | HPO4^2- | Tests whether higher water uptake improves or penalizes swing when phosphate is favored | PRIORITY-2 / uncertainty high |
| P5 | acrylic | Type I | macroporous, intermediate/connected | PO4^3- | Same architecture question with kinetics-favored phosphate | PRIORITY-2 / uncertainty high |
| P6 | acrylic | Type I | macroporous, intermediate/connected | P2O7^4- | High-charge phosphate derivative + high-water-uptake backbone | PRIORITY-2 / uncertainty high |
| P7 | styrenic | Type I | gel | HPO4^2- | Architecture control against P1 | CONTROL / informative |
| P8 | styrenic | Type I | gel | PO4^3- | Architecture control against P2 | CONTROL / informative |
| P9 | styrenic | Type II | macroporous | HPO4^2- | Deliberate challenge to known Type-II/carbonate preference | CHALLENGE / lower prior |
| P10 | styrenic | Type II | macroporous | PO4^3- | Tests whether tribasic phosphate can overcome Type-II preference | CHALLENGE / lower prior |
| P11 | styrenic | Type II | macroporous | P2O7^4- | Underexplored interaction candidate; evidence weaker | EXPLORATORY |
| C1 | matched to P1-P11 | matched | matched | CO3^2- | Required carbonate control on same architecture | CONTROL |

## First ranking before ML

This is an evidence-prior ranking only, not a performance ranking.

Tier A — test / map first:
- P1: Type I + good macroporous architecture + HPO4^2-
- P2: Type I + good macroporous architecture + PO4^3-
- P3: Type I + good macroporous architecture + P2O7^4-

Tier B — valuable interaction tests:
- P4-P6: acrylic Type I variants

Tier C — controls / falsification:
- P7-P8: gel controls
- P9-P11: Type II phosphate challenge branch
- C1: carbonate matched controls

## What must be resolved before atomistic screening

For every row selected for promotion:
1. map to one or more real, commercially available or synthetically realistic resin identities;
2. verify whether the exact resin × ion combination has already been experimentally reported;
3. extract comparable experimental features and moisture-swing outcomes from public datasets/literature;
4. estimate uncertainty / OOD status;
5. check practical ion-exchange loading, safety, procurement, and measurement feasibility;
6. retain a matched carbonate control.

No row should enter ALCHEMI merely because it looks chemically interesting.

## Immediate data task

Use MSA-ML and the literature only to populate this matrix with real resin identities and evidence:
- resin ID / trade name
- backbone
- Type I / Type II
- gel / macroporous
- BET / pore-volume / water-retention information when available
- tested counter-ions
- humidity conditions
- measured Qe / swing metrics
- whether each exact phosphate combination is known, missing, or out-of-distribution

The next output should be a gap map:
KNOWN → PARTIALLY KNOWN → UNTESTED/UNCLEAR → NOT FEASIBLE.

## Promotion rule

A phosphate combination becomes a computational candidate only when:
- exact material identity is defined;
- literature/data gap is real rather than assumed;
- ML or evidence ranking supports it;
- uncertainty is quantified;
- feasibility is acceptable;
- a matched control exists.

A computational candidate becomes a lab candidate only after the full discovery protocol gates are passed.

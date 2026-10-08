# Scale Engine V2 — Evidence & Feasibility Audit V1

Date: 2026-10-08

## Objective
Evaluate the current best-overall candidates for experimental usefulness. Novelty is **not** an exclusion rule.

## Evidence reviewed

### IRA900 platform
**Commercial/physical feasibility: HIGH.**
DuPont identifies AmberLite IRA900 Cl as a macroporous, styrene-divinylbenzene, Type I strong-base anion-exchange resin with trimethylammonium functional groups, shipped in chloride form, total exchange capacity >=1.0 eq/L, and broad service pH range.
Source: https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-AmberLite-IRA900-Cl-PDS-45-D01254-en.pdf

A 2026 Environmental Science & Technology study explicitly uses IRA900 as a model moisture-swing DAC sorbent and reports CO2 sorption dependence on relative humidity, CO2 partial pressure, and temperature.
Source: https://pubs.acs.org/doi/10.1021/acs.est.5c11862

A 2025 ACS Sustainable Chemistry & Engineering study reports that IRA900 counter-ions can be substituted by ion exchange and summarizes prior IRA900 moisture-swing work. It specifically notes that pyrophosphate-loaded IRA900 gave the highest CO2 swing capacity per ammonium site in a screened anion panel for a 20% -> 65% RH swing.
Source: https://pubs.acs.org/doi/10.1021/acssuschemeng.5c00227

### D201 platform
**Commercial/physical feasibility: MEDIUM-HIGH.**
Public technical datasheets describe D201 as a macroporous styrene-based strong-base anion exchanger with trimethylammonium functionality, chloride form, high exchange capacity, and 50–60% water retention.
Sources:
- https://www.lijiresin.com/files/d201.pdf
- https://www.chinaresin.com/product_detail/5.html

A vendor comparison describes D201 as structurally similar/equivalent in application class to AmberLite IRA900. This is useful for feasibility, but is not evidence that D201 duplicates IRA900 moisture-swing performance.
Source: https://www.bidragon.com/ion-exchange-resins/d201-macroporous-strong-base-anion-exchange-resin.html

## Candidate interpretation

| Candidate | ML rank signal | Direct moisture-swing evidence | Feasibility | Evidence status | Decision |
|---|---|---|---|---|---|
| IRA900 + PO4^3- | very high | direct IRA900 platform + phosphate-family evidence | high | STRONG | PROMOTE |
| IRA900 + HPO4^2- | very high | direct IRA900 platform; exact counter-ion observed in MSA | high | STRONG | PROMOTE |
| IRA900 + P2O7^4- | very high | direct published pyrophosphate moisture-swing signal | high | VERY STRONG | PROMOTE |
| IRA900 + CO3^2- | very high | standard/reference moisture-swing chemistry | high | VERY STRONG CONTROL | PROMOTE AS CONTROL |
| D201 + PO4^3- | high | weaker direct moisture-swing literature found; analogous resin class | medium-high | MODERATE | PROMOTE FOR CROSS-CHECK, NOT EQUIVALENCE CLAIM |

## Recommended experimental shortlist
1. **IRA900 + P2O7^4-** — strongest external moisture-swing signal among current finalists.
2. **IRA900 + PO4^3-** — strong overall computational score and experimentally plausible phosphate chemistry.
3. **IRA900 + HPO4^2-** — strong computational score and useful phosphate-speciation contrast.
4. **IRA900 + CO3^2-** — keep as matched control/reference.
5. **D201 + PO4^3-** — useful architecture-transfer candidate; requires more direct validation.

## Physics gate
Do **not** interpret a local atomistic model as a ranking replacement for the experimental evidence above.

Before a new ALCHEMI run:
- use an IRA900 Type-I local-site model, not the IRA910 Type-II motif;
- preserve counter-ion stoichiometry/charge neutrality;
- compare PO4 / HPO4 / P2O7 / carbonate with identical hydration and CO2 protocols;
- require geometry convergence before any energy comparison;
- use multiple seeds/hydration levels;
- keep D201 separate unless its local model is explicitly justified.

## Decision
**PASS to targeted physics/cross-check.**

This means the candidates are worth further computational/experimental evaluation. It does **not** mean that any candidate is newly discovered, superior in practice, or laboratory validated.

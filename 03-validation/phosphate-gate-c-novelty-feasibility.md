# Gate C — Novelty & Feasibility Check for HPO4 Candidates

Date: 2026-10-07

## Scope

Candidates entering this gate from uncertainty/OOD V2:
- IRA910 + HPO4^2-
- FPA54 + HPO4^2-

MN100 + HPO4^2- is deprioritized because the counterfactual ML screen predicted a much weaker signal.

## Candidate 1 — AmberLite IRA910 + HPO4^2-

### Material identity
- Strong-base anion exchange resin, Type II.
- Macroporous styrene-divinylbenzene matrix.
- Functional group reported as dimethylethanolamine-derived Type II quaternary ammonium.
- Commercially available in chloride form.

### Literature / novelty status
Targeted public searches found evidence for IRA910 as a real strong-base macroporous anion exchanger and for phosphate handling in ion-exchange contexts, but no obvious moisture-swing CO2 study of the exact IRA910 + HPO4^2- pair was identified in this search.

Status: **PARTIALLY KNOWN / exact moisture-swing HPO4 pair not found in targeted search**.

### Feasibility
- Real commercial resin: yes.
- Counter-ion exchange into phosphate-family species: chemically plausible for a strong-base anion exchanger.
- Matched carbonate and PO4 data exist in the MSA dataset.
- Exact HPO4 loading protocol and speciation control must be defined experimentally.

### Atomistic-model suitability
Current legacy ALCHEMI DQ-AER motif is **not** an exact model of IRA910.
IRA910 is Type II and should be represented by a benzyldimethylethanolammonium-like site on a styrenic environment, not by the prior inferred DQ-AER motif.

Decision: **PROMOTE to targeted atomistic model construction, not directly to legacy ALCHEMI scoring.**

## Candidate 2 — AmberLite FPA54 + HPO4^2-

### Material identity
- Weak-base anion exchange resin.
- Highly porous crosslinked phenol-formaldehyde matrix.
- Tertiary amine functionality.
- Free-base form as shipped.
- Product literature states high selectivity for phosphates.

### Literature / novelty status
Targeted public searches found FPA54 phosphate selectivity and general weak-base resin use, but no obvious moisture-swing CO2 study of the exact FPA54 + HPO4^2- pair was identified in this search.

Status: **PARTIALLY KNOWN / exact moisture-swing HPO4 pair not found in targeted search**.

### Feasibility
- Real commercial resin: yes.
- Phosphate interaction: explicitly plausible from product literature.
- But this is a **weak-base tertiary amine resin**, so HPO4 uptake/protonation chemistry is different from strong-base quaternary ammonium exchange.

### Atomistic-model suitability
The current legacy ALCHEMI strong-base/quaternary-ammonium setup is **not chemically appropriate** for FPA54 without building a different weak-base protonation model.

Decision: **DO NOT run the current strong-base ALCHEMI workflow on FPA54. Keep as experimental/alternative-chemistry branch.**

## Gate C decision

### Promote now
**IRA910 + HPO4^2-**

Reason:
- strongest ML information-value signal among current gaps;
- real commercial resin;
- matched carbonate and PO4 data exist;
- exact HPO4 moisture-swing pair not found in targeted search;
- compatible with a tractable Type II quaternary-ammonium local model once the site is rebuilt correctly.

### Hold
**FPA54 + HPO4^2-**

Reason:
- scientifically interesting and phosphate-selective;
- but weak-base tertiary-amine chemistry makes the current strong-base ALCHEMI model invalid.

## Required next test

Build a **new IRA910-specific Type II local motif** and validate that motif before any energetic comparison:
1. styrenic/benzyl environment;
2. Type II quaternary ammonium derived from dimethylethanolamine;
3. HPO4^2- versus matched CO3^2- control;
4. dry and hydrated states;
5. multiple starting geometries;
6. convergence checks;
7. no claim of real-material validation from a local motif alone.

Only after the IRA910-specific motif passes a sanity check should NVIDIA ALCHEMI be used for a targeted comparison.

## Guardrails

- "No obvious publication found" is not proof of novelty.
- FPA54 must not be simulated with the legacy strong-base motif.
- A local atomistic motif is not the whole resin.
- The next atomistic step is a falsification/support check, not a discovery claim.

# Evidence-first bibliography, 11 October 2026

This audit extends, **does not replace**, the existing `03-validation/scale-top-candidates-evidence-v1.md` and the Gate D chemical-identity checks. Literature is searched before expensive physics. No scale-to-million or new ALCHEMI calculation is authorized by this audit alone.

## Primary literature newly anchored to the source

| Source | DOI | Direct implication for V4 | Scope |
|---|---|---|---|
| Hegarty et al., *Expanding the Library of Ions for Moisture-Swing Carbon Capture* (2023) | [10.1021/acs.est.3c02543](https://doi.org/10.1021/acs.est.3c02543) | Existing work tested pyrophosphate, phosphate and related anions on ion-exchange resins. Pyrophosphate led on per-ion/charge normalization; dibasic phosphate led on mass-normalized swing; tribasic phosphate had kinetic advantages. Therefore no novelty claim for these ions or the previously tested IRA900+P2O7 formulation | Prior art, specific studied ion family; exact-pair evidence corroborated by Wang 2025 |
| Wang et al., *Investigation of Moisture Swing Adsorbents for Direct Air Capture by Dynamic Breakthrough Studies* (2025) | [10.1021/acssuschemeng.5c00227](https://doi.org/10.1021/acssuschemeng.5c00227) | Direct IRA900-C breakthrough work, CO2 working-capacity and internal mass-transfer/kinetic effects. Carbonate control must use comparable RH, temperature, CO2 and time/geometry conditions | Measured IRA900-C, not P2O7 experiment |
| Hegarty et al., *Structure–Property Relationships for Moisture-Swing Direct Air Capture* (2026) | [10.1021/acs.est.5c16820](https://doi.org/10.1021/acs.est.5c16820) | Ten commercial resins: the counter-ion interacts with Type I/II ammonium identity, pore connectivity, polymer backbone and kinetics; Type I vs Type II cannot share an invented one-size-fits-all atomistic structure | Family / mechanism, not every exact pair |
| *CO2 Sorption in Moisture Swing Anion Exchange Resins for Direct Air Capture: Experimental Isotherm Determination and Modeling* (2026) | [10.1021/acs.est.5c11862](https://doi.org/10.1021/acs.est.5c11862) | Measurement strategy for IRA900 should compare comparable temperature, humidity and CO2 pressure; static single-energy proxy insufficient | Platform/measurement context |
| *Active Learning-Guided Optimization of Moisture Swing Adsorption for Direct Air Capture* (26 Sep 2026) | [10.1021/acs.est.6c07244](https://doi.org/10.1021/acs.est.6c07244) | Published active-learning framework is our prior methodological comparator. 290 data points, 201 conditions, 20 commercial resins. Its `PYU-pub/MSA-ML` code/data are **not external independent validation** of a model trained on that same dataset | Dataset/method prior art; examine exact repository license separately |
| Song et al., *Moisture Swing Ion-Exchange Resin-PO4 Sorbent for Reversible CO2 Capture from Ambient Air* (2019) | [10.1021/acs.energyfuels.9b00863](https://doi.org/10.1021/acs.energyfuels.9b00863) | PO4-based moisture-swing capture is established prior art; exact chemistry/resin/processing distinctions still matter | Phosphate-family prior art |
| DuPont IRA900 Cl product datasheet | [manufacturer PDF](https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-AmberLite-IRA900-Cl-PDS-45-D01254-en.pdf) | Commercial Type I macroporous resin identity is documented, but purchase and counter-ion exchange feasibility must be established separately | Manufacturer product identity, not adsorption proof |

The scope/claims above are recorded separately in `chemistry/v4/literature/curated_evidence.json` with URLs, DOI, publication date and explicit reviewed status. Historic D201 datasheet is retained but marked `PREVIOUS_AUDIT_PENDING_REFRESH`; product equivalence to IRA900 is NOT demonstrated.

## What this changes

- The discovery proposition is **better pair selection, robust protocol and costly-experiment reduction**, not rediscovery of phosphate/pyrophosphate.
- Reuse the existing MSA-ML dataset for baseline reproduction, but evaluate against truly separate future lab experiments before independent external-validation claims.
- Use matched carbonate controls and compare *moisture-swing working capacity*, adsorption/desorption kinetics, loading-normalized and mass-normalized metrics separately.
- Type I IRA900, Type II IRA910 and weak-base variants must have *chemically faithful* different local atomistic models.
- Retain QMPR and other unreviewed pair hypotheses in HOLD pending named manufacturer/molecular evidence.
- A publication missing from one API query does **not** prove novelty. No DOI metadata search result alone upgrades Gate D to PASS.
- Full-text reuse and redistribution are excluded. Only factual citation metadata/curated brief claims are committed; the dataset's licensing requires its own review before commercial use.

## Automatic refresh and review queue

```bash
python chemistry/v4/test_literature.py
python chemistry/v4/literature.py --rows 6
```

The V4 workflow runs these in a standalone bibliography job **before** Gate D and stores `literature.json` and `literature.md`. The updater executes a bounded Crossref public metadata search across six chemical, ML, kinetic and resin queries, deduplicates by DOI, filters out publications after the as-of date, and emits newly found metadata as `NEW_METADATA_UNREVIEWED`. Network error is visible as `DEGRADED_*`; locally curated references remain available. Human review is required to add a verified citation to the curated file.

## Next questions before a million-scale campaign

1. Review which exact prior-paper material/candidate pairs are genuinely contained in the underlying data vs only inferred from descriptors.
2. Confirm the licence and version/commit of MSA-ML raw Excel before any commercial redistribution.
3. Resolve resin-specific full-polymer representativeness, pore connectivity and charge/stoichiometry for physics finalists.
4. Keep carbonate and appropriate phosphate reference in every downstream lab protocol; test kinetics and repeated cycles, not only a static energy shift.

No material is claimed experimentally validated by this software audit.

# GFN1 V4 — bounded independent closure (11 October 2026)

## Verified starting point
- Canonical V4 historical run [38095586192](https://github.com/smileflowk-dotcom/athar-global/actions/runs/38095586192): independent GFN1 PO4 had 3/1/0/0 valid pairs at 0/3/6/9 H2O (HOLD). AIMNet2 result is independent, not a substitute for GFN1.
- [38168524639](https://github.com/smileflowk-dotcom/athar-global/actions/runs/38168524639), GFN1-only PO4 3/6/9 H2O, 3 near/far seeds per state: 3/3 at each point, with original fmax force acceptance <= 0.40 eV/A and continued FIRE to at most 240 steps. No carbonate and no dry state; diagnostic only, artifact 11714382108.
- [38175588297](https://github.com/smileflowk-dotcom/athar-global/actions/runs/38175588297): independently recomputed PO4 and matched CO3 3/6/9 H2O, two seeds, 2/2 near/far pairs per hydration per ion. Required continuation occurred on failing initial forces; fmax optimizer target 0.30 and acceptance <= 0.40 were unchanged. Source trace, pre/post forces, energies, pair results, pip freeze and solver log stored in GitHub artifact **11716871740**, 90-day retention.
- [38178109536](https://github.com/smileflowk-dotcom/athar-global/actions/runs/38178109536) fails *before physics*, due to a brittle exact text-anchor source check (`Existing force threshold changed: STOP`). This is a **preflight string mismatch**, NOT new evidence of GFN1 chemistry failure. Do not rerun that duplicate branch.

## Diagnosis and bounded correction
The initial 120-step optimizer budget is insufficient for several microhydrated configurations: at 3/6/9 H2O the successful matched control diagnostic observed many initially rejected near/far forces, then accepted them after **continuing the same FIRE optimizer** (at most 120 additional steps). The chemistry/force criteria, site, charge, GFN1 method, deterministic seeds and matched carbonate are unchanged. This addresses convergence exhaustion, **not** model adequacy or true adsorption.

## Only remaining test
- Compute exclusively **0 H2O** on PO4 and matched CO3, seeds 0/1, near/far; identical instrumented V4 GFN1 continuation.
- Download and verify the successful prior 3/6/9 raw per-state pairs (not recomputed) and pin the previously used GFN1 scientific package versions.
- Verify exactly 2 valid accepted pairs at **all 0/3/6/9** for both species, plus a **positive 0→9 PO4 median-delta shift**. Retain raw force/energy diagnostics, generated source, SHA256 hashes, dependencies and complete matched report.
- Always retain HOLD unless all checks succeed. Even success labels only `READY_FOR_SCIENTIFIC_REVIEW`, never a material discovery or lab result. Do not change V4 cache or launch scale engine.

## Limits
Local finite fragment + microhydration proxy; not relative humidity, sorption free energy, experimentally realized IRA900 resin nor validation. A future full matched 4-hydration independent replication and lab investigation may still be required before any claim. Artifact retention on GitHub is time-limited; the final audit embeds re-used numeric rows and checksums for traceability.

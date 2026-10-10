# Chemistry Discovery Engine V4 (within ATHAR Global)

This is a **reusable orchestration layer** with an initial `co2_moisture_swing` adapter. It is **not** the future dedicated chemistry repository; do not create or split repositories yet.

## What's reused
- `scale/candidate_factory_v1.py`: existing public MSA dataset + resin evidence.
- `scale/candidate_factory_v2.py`: existing counter-ion descriptors.
- `scale/batch_scorer_v1.py`: existing LightGBM model definition.
- `scripts/autopilot_physics_v1.py`: existing AIMNet2 / independent GFN1 physics implementation; V4 tightens GFN1 acceptance.
- Existing GitHub Actions artifact evidence from runs 37998848865 and 38003905128, plus the IRA900/P2O7 and QMPR2/SO3 historical checks.

## Scientifically gated end-to-end workflow (no reset)

The authoritative scientific order remains:
**READ PRIOR EVIDENCE → DATA → GROUPED ML → UNCERTAINTY/OOD →
FOUR-MODE ACTIVE LEARNING → LITERATURE/CHEMICAL FEASIBILITY →
NVIDIA ALCHEMI/AIMNet2 → INDEPENDENT GFN1-xTB → LAB.**

One GitHub Actions entrypoint: `.github/workflows/chemistry-discovery-v4.yml`.

1. **Streamed candidate factory**: 1M default / 3M supported **virtual operating conditions**, not new molecules. Resume checkpoints and retain the best **distinct** experimental operating conditions by resin+ion. No 1M-row in-memory table.
2. **Real staged funnel**: `10k → 1k → 100 → 20` with controlled chemical-resin diversity, quotas and reproducible CSVs. Every stage reports its *actual* observed count: do not invent 10K molecules if fewer distinct conditions exist.
3. **Gate A — experimental validity**: holdout by `expID` against a naive training-mean Qe predictor, across two seeds. If the model cannot outperform baseline on experimental groups, no new atomistic jobs are scheduled.
4. **Gate B — strict generalization and uncertainty**: leave-one-resin-out with **group-bootstrap multiplicities preserved**. Negative or highly uncertain prediction remains HOLD. The current OOD is a bounded descriptor proxy only.
5. **Gate C — active learning**: exploitation, exploration, balanced performance/uncertainty, and diversity with an explicit carbonate control and historical phosphate priority. Acquisition reason is stored for every selected candidate; only 5 feasibility-passing finalists enter expensive physics.
6. **Gate D — literature and feasibility**: automated deterministic checks against the previously audited `03-validation/scale-top-candidates-evidence-v1.json` and primary links in the audit Markdown; validate counter-ion formal charge, charge-neutral site stoichiometry with RDKit, and reject unsupported resin-site equivalence. Missing independent literature/procurement/safety verification remains **HOLD**, never an invented novelty or synthesis PASS.
7. **Gate E — physics**: reuse prior named artifact evidence (including negative results). Strict NVIDIA ALCHEMI/AIMNet2 checks on 0/3/6/9 explicit waters and multiple seeds; independent GFN1-xTB using those *same water counts*. Require ≥2 converged matched near/far pairs **for the target AND carbonate** at each hydration. Two-water historical GFN1 PASS is now **PRECHECK**, not full robust PASS.
8. **Automated report**: phase-specific machine-readable JSON and CSV artifacts + Markdown, with gating A–D, evidence provenance and no lab validation claim. **No experimental validation occurs on GitHub Actions.**

ALCHEMI is used only when the physics planner identifies an admissible missing atomistic test. The quick smoke explicitly sets physics budget zero, so ALCHEMI is **not** invoked in smoke runs.

### Current limitation
The Gate D audit is automated *over already reviewed evidence*, but it is not a live systematic literature search or chemical safety approval. New pairs require scientist review and additional references. We never mistake vendor similarity for validated molecular geometry. The candidate factory is initially specific to the CO₂ moisture-swing case; future general chemistry gets its own adapters.


## Launch
Actions -> **Chemistry Discovery V4** -> Run workflow. Default full mode: 1,000,000 virtual configurations, chunk size 5,000, physics budget 2. The push event runs a **small 1,500-configuration smoke** with physics budget zero; no substantial CPU computation is triggered by code changes.

## Performance contract
- Million-scale **coarse scored configurations**, not a million unique molecules or million independently validated materials.
- One model and streaming batches for broad screening; heavier ensemble only for top shortlists.
- Reuse existing experimental dataset, code and permitted cached artifacts.
- Time per run depends on GitHub runner capacity; no fixed speed promise.
- CPU-only physics jobs can still be slow. GPU must not be claimed without CUDA evidence.
- Do not silently recompute prior robust evidence, and do not equate an old V1 quick precheck with a stringent full robustness test.

## Scientific limitations
- Initial adapter only knows the CO2 resin/counter-ion family and available local-site geometries. Unsupported chemistry is held out, not simulated with invented structures.
- Descriptors for newly proposed counter-ions are approximations; chemistry feasibility review is still necessary.
- A positive microhydration binding-energy shift is a *proxy* for a moisture response, **not** actual CO2 capture capacity, selectivity, kinetics, regeneration energy, or material stability.
- `PASS` in V1 original GFN1 only checked finite forces, so V4 treats it as advisory.
- V4 targeted physics automatically runs four hydration levels and three geometry trials; completed checks remain computational proxies requiring independent experimental measurement.
- No one is allowed to claim discovery, lab validation, or engineering performance from this workflow alone.

## Future chemistry expansion
The intended future independent engine must use adapter contracts for `domain`, `candidate schema`, `data provenance`, `simulator`, `independent verifier`, and `experimental gate`. For now only `co2_moisture_swing` is implemented, without general-chemistry claims.

## Output
Artifacts: `v4-discovery`, `v4-plan`, `v4-prior-evidence`, `v4-physics-*` (if any), and `v4-final`. The final decision is `chemistry/v4/output/final.json`, with an accompanying readable `final.md`.

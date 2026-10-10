# Chemistry Discovery Engine V4 (within ATHAR Global)

This is a **reusable orchestration layer** with an initial `co2_moisture_swing` adapter. It is **not** the future dedicated chemistry repository; do not create or split repositories yet.

## What's reused
- `scale/candidate_factory_v1.py`: existing public MSA dataset + resin evidence.
- `scale/candidate_factory_v2.py`: existing counter-ion descriptors.
- `scale/batch_scorer_v1.py`: existing LightGBM model definition.
- `scripts/autopilot_physics_v1.py`: existing AIMNet2 / independent GFN1 physics implementation; V4 tightens GFN1 acceptance.
- Existing GitHub Actions artifact evidence from runs 37998848865 and 38003905128.

## End-to-end workflow
`.github/workflows/chemistry-discovery-v4.yml` is one GitHub Actions entry point.

1. **Streamed coarse ranking**: evaluate up to 3,000,000 *virtual operating configurations* in limited-memory batches using a single ML model. Report actual count processed. No million-row frame is retained.
2. **Strict shortlist ML**: reuse the existing experiment-grouped, leave-one-resin-out paradigm with an ensemble. This second, expensive stage runs only on few candidates.
3. **Evidence reuse**: selectively download the named prior Actions artifacts. Check candidate + method + result, distinguishing a generic job PASS from a physically positive hydration shift. Stronger robust evidence supersedes weaker V1 prechecks; weak prechecks do not count as final physics PASS.
4. **Bounded physics**: run only selected missing GFN1-xTB / AIMNet2 checks, in parallel, constrained by `physics_budget`. Source models are retained, not rewritten. Updated GFN1 convergence rejection is required.
5. **Automated decision**: save machine-readable JSON and a concise Markdown candidate/evidence table. Unfinished checks receive HOLD, not a fictional PASS.

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
- A V4 quick precheck remains preliminary. Full 4-point multi-seed robust confirmation and independent measurement remain explicit gates.
- No one is allowed to claim discovery, lab validation, or engineering performance from this workflow alone.

## Future chemistry expansion
The intended future independent engine must use adapter contracts for `domain`, `candidate schema`, `data provenance`, `simulator`, `independent verifier`, and `experimental gate`. For now only `co2_moisture_swing` is implemented, without general-chemistry claims.

## Output
Artifacts: `v4-discovery`, `v4-plan`, `v4-prior-evidence`, `v4-physics-*` (if any), and `v4-final`. The final decision is `chemistry/v4/output/final.json`, with an accompanying readable `final.md`.

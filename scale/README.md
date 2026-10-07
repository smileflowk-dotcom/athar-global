# Scale Engine V0

Purpose: industrialize the validated pieces of the moisture-swing CO2 screening workflow without claiming material discovery.

Funnel:
10K virtual/known configurations -> 1K ML/OOD/uncertainty -> 100 diversity/novelty/feasibility -> 20 heavier checks -> 5 ALCHEMI + independent cross-check -> 1-3 lab candidates.

## Compute policy
- Cheap filters and ML first.
- NVIDIA/Nebius Nemotron is an audit/reasoning layer, never physicochemical proof.
- NVIDIA ALCHEMI/AIMNet2 is reserved for finalists, never for the full 10K pool.
- GitHub Actions currently demonstrates CPU ALCHEMI. GPU execution must be explicitly evidenced by runtime logs before it is called GPU.
- Every candidate retains provenance and uncertainty.
- No "discovered", "validated", or "best" claim before independent cross-check + lab.

## Current NVIDIA state
- Nebius Token Factory + NVIDIA Nemotron route: previously validated in GitHub Actions.
- nvalchemi-toolkit/AIMNet2/FIRE2: validated.
- Current GitHub-hosted ALCHEMI runs: CPU, CUDA unavailable.
- Direct NVIDIA GPU/runtime capacity must be attached as a separate execution backend before scaling expensive atomistic jobs.

## Next implementation order
1. Candidate schema (this folder).
2. Candidate factory from observed/traceable chemistry only.
3. Batch scorer using existing Gate A/B logic.
4. Diversity-preserving 10K->1K->100 gates.
5. Evidence/feasibility enrichment.
6. Parallel 20->5 physics runner.
7. Independent cross-check and lab packet.

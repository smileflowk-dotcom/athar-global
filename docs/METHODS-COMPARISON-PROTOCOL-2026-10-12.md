# ATHAR V4 — controlled comparative methods pilot

**Draft protocol, not a paper result or public disclosure approval (12 Oct 2026).**

## Scientific question
Does the existing ATHAR V4 LightGBM configuration improve prediction of *measured Qe* over a training-mean model and a fixed-parameter example drawn from the MSA-ML authors' own public script, using identical held-out experimental groups? Report negative results as well as positive ones.

## Data and attribution
- Third-party experimental data: PYU-pub/MSA-ML, data/MSA data.xlsx, pinned revision 8324361b737324b6631c56018c3366554536e581. Commercial/data redistribution rights UNVERIFIED. Data stays transient in memory.
- Reference implementation: PYU-pub/MSA-ML code/common.py and code/Run LGB.py (note the public README uses a different filename). Fixed example LightGBM settings are independently encoded, without copying the upstream script.
- ATHAR model: existing checked-in scale/batch_scorer_v1.py::_model. No new model family or third-party code added.
- Primary scientific context: Hegarty 2023, DOI 10.1021/acs.est.3c02543; Hegarty 2026, DOI 10.1021/acs.est.5c16820; MSA-ML active-learning study, DOI 10.1021/acs.est.6c07244.
- The experiment must be labeled *fixed-parameter proxy comparison*, NOT a replication of published MSA-ML scores.

## Planned and implemented experiments
1. Require measured Qe, expID, resin and feature schema. Exclude counterfactual synthetic candidates as ground truth.
2. Five global GroupShuffleSplit seeds: 5, 10, 15, 20, 25; 20% experimental-group test holdouts. All rows of the same expID must remain together even across source classes.
3. Impute missing numerical descriptors with training-partition medians only. All four model arms evaluate on exactly the same test examples: training mean, MSA-ML-script fixed example LightGBM, ATHAR fixed LightGBM, and ATHAR LightGBM with base_type ablated.
4. Report mean and SD across five seeds for RMSE, MAE, R². These are overlapping holdouts, NOT independent confirmatory replications or inferential significance tests.
5. Secondary test: leave-one-resin-out for each known resin with expID overlaps excluded from training; report macro-averaged RMSE/MAE. Underfilled coverage is explicit, not disguised as generalization.

## Differences from the authors' published protocol
The upstream MSA-ML implementation uses per-source outer splits, inner grouped cross-validation, ten seeds and 100 randomized hyperparameter search iterations. This pilot uses fixed published-example hyperparameters and a GLOBAL expID split for leakage control. It therefore cannot honestly be represented as reproducing the authors' published benchmark, nor as showing superiority over their optimally tuned model.

The ATHAR ablation tests only whether the base_type feature changes Qe prediction. It does not test the incremental value of uncertainty/OOD, active learning, chemistry gate D or independent AIMNet2/GFN1 checks. These mechanisms would require a separate, pre-specified prospective selection study.

## What this study does NOT establish
- No experimental demonstration of new sorbents or improved CO2 uptake.
- No independent dataset for predicting actual low-vs-high humidity working capacity.
- No measured reduction in laboratory cost or selection hit rate.
- No evidence of novel phosphate or pyrophosphate anion chemistry: prior art.
- No claim of patentability or research article originality.

## Confidentiality and publication
This repository, all branches and draft PRs are PUBLIC. Only the aggregate-only benchmark source, synthetic tests and this protocol may be included. Do not commit raw third-party workbook, per-sample predictions, trained weights, unpublished candidate rankings, personal contacts or new private lab results. **Execute the real-data evaluation only in an approved private workspace and review any new results for IP and dataset permissions BEFORE any public disclosure.**

## Reproduce privately (no Docker, no GPU)
With Python, numpy, pandas, openpyxl, scikit-learn and lightgbm in an authorized environment:

    python scripts/test_public_msa_methods_benchmark_v1.py
    python scripts/public_msa_methods_benchmark_v1.py --with-loro --output /tmp/athar-methods-v1

Outputs: benchmark.json and benchmark.md containing only aggregate/per-fold metrics, environment versions and dataset SHA256. The public CI only tests synthetic fixtures, never fetches the real workbook.

## Prerequisites before submission
Validate all DOI claims and source locators against the full papers, record each factual claim's evidence, pin ATHAR commit + dependencies, reproduce the full published MSA-ML nested-CV baseline (or retain the weaker claim), establish data-license permissions, obtain an outside specialist review, and conduct independent matched-carbonate laboratory tests before describing physical material improvements. Human owners must approve disclosure.

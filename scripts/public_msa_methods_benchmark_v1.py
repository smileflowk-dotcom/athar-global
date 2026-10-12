#!/usr/bin/env python3
"""Aggregate-only MSA-ML-script vs ATHAR V4 measured-Qe benchmark.

Not a reproduction of published MSA-ML scores, new CO2 sorbent discovery,
or experimental validation. Outputs contain no workbook or row-level predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import sys
import urllib.request
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scale.batch_scorer_v1 import _model as athar_model, FEATURE_CAT, FEATURE_NUM, FEATURES

SOURCE_REF = "8324361b737324b6631c56018c3366554536e581"
SOURCE_URL = f"https://raw.githubusercontent.com/PYU-pub/MSA-ML/{SOURCE_REF}/data/MSA%20data.xlsx"
SOURCE_MODEL = f"https://github.com/PYU-pub/MSA-ML/blob/{SOURCE_REF}/code/Run%20LGB.py"
SOURCE_METHOD = f"https://github.com/PYU-pub/MSA-ML/blob/{SOURCE_REF}/code/common.py"
REF_CAT = ["pore", "matrix", "FG"]
REF_NUM = list(FEATURE_NUM)
REF_FEATURES = REF_CAT + REF_NUM
SEEDS = (5, 10, 15, 20, 25)
ARMS = ("naive_training_mean", "msa_ml_script_fixed_lgb",
        "athar_v4_fixed_lgb", "athar_v4_no_base_type")


def load_public_dataset():
    """Fetch an explicitly pinned upstream workbook transiently into memory."""
    with urllib.request.urlopen(SOURCE_URL, timeout=40) as response:
        raw = response.read(5_000_001)
    if not raw or len(raw) > 5_000_000:
        raise ValueError("Dataset download failed safety check")
    return pd.read_excel(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()


def validate_schema(df):
    required = set(FEATURES) | {"Qe", "expID", "resin", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Unsupported dataset schema: " + ", ".join(missing))
    d = df.dropna(subset=["Qe", "expID", "resin"]).copy()
    d["Qe"] = pd.to_numeric(d["Qe"], errors="coerce")
    d = d.dropna(subset=["Qe"])
    if len(d) < 50 or d["expID"].nunique() < 10 or d["resin"].nunique() < 3:
        raise ValueError("Insufficient independent measured data")
    if d["resin"].astype(str).str.strip().eq("").any():
        raise ValueError("Unidentified resin")
    if ("provenance_source_type" in d.columns and
        d["provenance_source_type"].astype(str).str.contains(
            "virtual_counterfactual").any()):
        raise ValueError("Virtual configurations are not measured ground truth")
    return d.reset_index(drop=True)


def fold_clean(train, test):
    """Numerical imputation uses only the training partition."""
    tr, te = train.copy(), test.copy()
    for c in FEATURE_CAT:
        tr[c] = tr[c].fillna("unknown").astype(str)
        te[c] = te[c].fillna("unknown").astype(str)
    for c in FEATURE_NUM:
        x = pd.to_numeric(tr[c], errors="coerce")
        y = pd.to_numeric(te[c], errors="coerce")
        m = x.median()
        if not np.isfinite(m):
            raise ValueError("Missing numeric training distribution: " + c)
        tr[c], te[c] = x.fillna(m), y.fillna(m)
    return tr, te


def msa_lgb(seed):
    """Reimplements fixed example parameters, NOT author's nested CV search."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), REF_CAT),
        ("num", StandardScaler(), REF_NUM),
    ])
    params = dict(objective="regression", n_estimators=150, max_depth=6,
                  learning_rate=.07, min_child_samples=6, subsample=.8,
                  colsample_bytree=.8, reg_alpha=.5, reg_lambda=3,
                  random_state=seed, n_jobs=1, verbosity=-1)
    return Pipeline([("pre", pre), ("lgb", LGBMRegressor(**params))])


def metrics(y, predictions):
    a, p = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    if len(a) != len(p) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Invalid metric inputs")
    return {"n_test": int(len(a)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "mae": float(mean_absolute_error(a, p)),
            "r2": float(r2_score(a, p)) if len(a) > 1 else None}


def fit_all(train, test, seed):
    tr, te = fold_clean(train, test)
    ytr, yte = tr["Qe"].to_numpy(dtype=float), te["Qe"].to_numpy(dtype=float)
    ref = msa_lgb(seed)
    ref.fit(tr[REF_FEATURES], ytr)
    native = athar_model(seed)
    native.set_params(model__n_jobs=1)
    native.fit(tr[FEATURES], ytr)
    tr_ab, te_ab = tr.copy(), te.copy()
    tr_ab["base_type"] = "ABLATION_CONSTANT"
    te_ab["base_type"] = "ABLATION_CONSTANT"
    ablated = athar_model(seed)
    ablated.set_params(model__n_jobs=1)
    ablated.fit(tr_ab[FEATURES], ytr)
    return {
        "naive_training_mean": metrics(yte, np.full(len(te), ytr.mean())),
        "msa_ml_script_fixed_lgb": metrics(yte, ref.predict(te[REF_FEATURES])),
        "athar_v4_fixed_lgb": metrics(yte, native.predict(te[FEATURES])),
        "athar_v4_no_base_type": metrics(yte, ablated.predict(te_ab[FEATURES])),
    }


def benchmark(df, seeds=SEEDS, with_loro=False):
    d = validate_schema(df)
    group = d["expID"].astype(str)
    runs = []
    for seed in seeds:
        split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
        itr, ite = next(split.split(d, d["Qe"], groups=group))
        if set(group.iloc[itr]) & set(group.iloc[ite]):
            raise RuntimeError("Experimental group leakage")
        runs.append({"seed": seed, "train_groups": int(group.iloc[itr].nunique()),
                     "test_groups": int(group.iloc[ite].nunique()),
                     "metrics": fit_all(d.iloc[itr], d.iloc[ite], seed)})
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {k: {
            "mean": float(np.mean([z["metrics"][arm][k] for z in runs])),
            "sd": float(np.std([z["metrics"][arm][k] for z in runs], ddof=1))
                  if len(runs) > 1 else 0.
        } for k in ("rmse", "mae", "r2")}
    results = {
        "status": "COMPUTATIONAL_METHODS_PILOT_NOT_PEER_REVIEWED",
        "task": "measured Qe (not moisture-swing working capacity)",
        "dataset_source": SOURCE_URL, "source_commit": SOURCE_REF,
        "reference_method_source": SOURCE_METHOD,
        "reference_hyperparameters_source": SOURCE_MODEL,
        "n_observations": len(d), "n_experimental_groups": int(group.nunique()),
        "n_resins": int(d["resin"].nunique()),
        "split_policy": "global expID holdout across all source classes",
        "protocol_difference": (
            "NOT the original MSA-ML authors' per-source split, nested grouped CV "
            "or 100-iteration tuning. Reference uses fixed public script examples."),
        "seed_results": runs, "aggregate": aggregate,
        "limitations": [
            "No external validation dataset", "Measured Qe is not moisture-swing capacity",
            "No evaluation of active-learning experimental hit rate",
            "No demonstrated material discovery or reduction in laboratory cost",
            "No novelty or patentability conclusion"
        ],
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__,
                        "lightgbm": lightgbm.__version__}
    }
    if with_loro:
        resin_rows = []
        for n, resin in enumerate(sorted(d["resin"].astype(str).unique())):
            test = d[d["resin"].astype(str) == resin]
            train = d[d["resin"].astype(str) != resin]
            train = train[~train["expID"].astype(str).isin(test["expID"].astype(str))]
            if len(test) == 0 or len(train) < 30 or train["expID"].nunique() < 4:
                continue
            # No per-row outputs, only summary performance.
            resin_rows.append({"resin_index": n + 1, "n_test": int(len(test)),
                               "metrics": fit_all(train, test, 123 + n)})
        results["leave_one_resin_out"] = {
            "status": "COMPLETE" if len(resin_rows) == d["resin"].nunique() else "INCOMPLETE",
            "count": len(resin_rows), "expected_count": int(d["resin"].nunique()),
            "summary": {
                arm: {m: float(np.mean([z["metrics"][arm][m] for z in resin_rows]))
                      for m in ("rmse", "mae")} for arm in ARMS
            } if resin_rows else {}
        }
    return results


def markdown_summary(r):
    lines = [
        "# ATHAR V4 vs MSA-ML-script: controlled methods pilot", "",
        "**Not paper replication, independent measurement, or new material validation.**",
        "", f"Rows: {r['n_observations']}; experimental groups: {r['n_experimental_groups']}; "
            f"resins: {r['n_resins']}", "",
        "Target: Qe on globally held-out expID groups (not moisture-swing capacity).", "",
        "| Arm | RMSE mean ± SD | MAE mean ± SD | R² mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for arm, row in r["aggregate"].items():
        fmt = lambda k: f"{row[k]['mean']:.4f} ± {row[k]['sd']:.4f}"
        lines.append(f"| {arm} | {fmt('rmse')} | {fmt('mae')} | {fmt('r2')} |")
    lines += [
        "", "The upstream reference uses *fixed example* LightGBM hyperparameters, "
        "NOT the authors' 100-iteration nested-CV tuning. Both pipelines were "
        "evaluated on matched held-out groups.", "",
        "**No measured capture-capacity, lab-cost or novelty improvement is implied.**"
    ]
    if "leave_one_resin_out" in r:
        x = r["leave_one_resin_out"]
        lines += ["", f"LORO: {x['status']}, {x['count']}/{x['expected_count']} resins."]
    lines += [
        "", "Original source code and data: PYU-pub/MSA-ML (permission for "
        "commercial redistribution NOT established). No dataset, per-row "
        "predictions or trained weights included in the output."
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/athar-methods-v1")
    p.add_argument("--with-loro", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    df, sha = load_public_dataset()
    r = benchmark(df, with_loro=a.with_loro)
    r["upstream_raw_sha256"] = sha
    (out / "benchmark.json").write_text(json.dumps(r, indent=2) + "\n")
    summary = markdown_summary(r)
    (out / "benchmark.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()

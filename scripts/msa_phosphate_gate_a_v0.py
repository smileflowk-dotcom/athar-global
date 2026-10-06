#!/usr/bin/env python3
import json, os, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

OUT=Path("03-validation")
OUT.mkdir(exist_ok=True)
DATA=Path("/tmp/MSA-data.xlsx")
URL="https://raw.githubusercontent.com/PYU-pub/MSA-ML/main/data/MSA%20data.xlsx"
urllib.request.urlretrieve(URL, DATA)

df=pd.read_excel(DATA)
cols=list(df.columns)

cat=[c for c in ["pore","matrix","FG"] if c in df.columns]
num=[c for c in ["water_retention","BET","Vp","N%","FG:RU","IE-capacity","anion_charge","anion-pKa","T","humidity","Cini","M/V"] if c in df.columns]
features=cat+num
required=["Qe","expID"]
missing=[c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

work=df.dropna(subset=["Qe","expID"]).copy()
for c in cat:
    work[c]=work[c].astype(str)
for c in num:
    work[c]=pd.to_numeric(work[c], errors="coerce")

# conservative imputation by median for numeric, explicit unknown for categorical
for c in num:
    work[c]=work[c].fillna(work[c].median())
for c in cat:
    work[c]=work[c].fillna("unknown")

groups=work["expID"]
n_splits=min(5, int(groups.nunique()))
if n_splits < 3:
    raise SystemExit("Not enough experimental groups for grouped validation")

pre=ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num),
])

def model_pipe():
    return Pipeline([
        ("pre", pre),
        ("model", LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            max_depth=-1, subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbosity=-1
        ))
    ])

gkf=GroupKFold(n_splits=n_splits)
pred=np.full(len(work), np.nan)
dummy_pred=np.full(len(work), np.nan)
folds=[]
X=work[features]
y=work["Qe"].astype(float).values
for i,(tr,te) in enumerate(gkf.split(X,y,groups=groups),1):
    m=model_pipe()
    m.fit(X.iloc[tr], y[tr])
    p=m.predict(X.iloc[te])
    pred[te]=p
    d=DummyRegressor(strategy="mean").fit(np.zeros((len(tr),1)), y[tr])
    dp=d.predict(np.zeros((len(te),1))
    )
    dummy_pred[te]=dp
    folds.append({
        "fold":i,
        "n_train":len(tr),"n_test":len(te),
        "r2":float(r2_score(y[te],p)),
        "mae":float(mean_absolute_error(y[te],p)),
        "rmse":float(mean_squared_error(y[te],p)**0.5),
    })

overall={
    "r2":float(r2_score(y,pred)),
    "mae":float(mean_absolute_error(y,pred)),
    "rmse":float(mean_squared_error(y,pred)**0.5),
}
dummy={
    "r2":float(r2_score(y,dummy_pred)),
    "mae":float(mean_absolute_error(y,dummy_pred)),
    "rmse":float(mean_squared_error(y,dummy_pred)**0.5),
}

# Evidence coverage audit: no guessed chemistry.
anion_col="anion" if "anion" in work.columns else None
resin_col="resin" if "resin" in work.columns else ("resinID" if "resinID" in work.columns else None)
phosphate_rows=pd.DataFrame()
if anion_col:
    mask=work[anion_col].astype(str).str.contains("phosph|HPO4|PO4|P2O7|pyroph", case=False, regex=True, na=False)
    phosphate_rows=work.loc[mask].copy()

coverage={
    "dataset_rows":int(len(work)),
    "experimental_groups":int(groups.nunique()),
    "columns":cols,
    "features_used":features,
    "anion_column":anion_col,
    "resin_column":resin_col,
    "phosphate_rows":int(len(phosphate_rows)),
}
if anion_col:
    coverage["anion_values"]=sorted(work[anion_col].dropna().astype(str).unique().tolist())
    coverage["phosphate_anion_values"]=sorted(phosphate_rows[anion_col].dropna().astype(str).unique().tolist())
if resin_col and len(phosphate_rows):
    coverage["phosphate_resins"]=sorted(phosphate_rows[resin_col].dropna().astype(str).unique().tolist())

# Only descriptive stats; do not infer causal or moisture-swing ranking from single-row Qe.
phosphate_summary=[]
if len(phosphate_rows) and anion_col:
    group_cols=[anion_col]+([resin_col] if resin_col else [])
    g=phosphate_rows.groupby(group_cols, dropna=False)["Qe"].agg(["count","mean","median","std"]).reset_index()
    phosphate_summary=g.to_dict(orient="records")

report={
    "status":"PASS" if overall["r2"]>dummy["r2"] else "FAIL",
    "test":"Gate A grouped predictive credibility + phosphate data coverage audit",
    "source_dataset":"PYU-pub/MSA-ML data/MSA data.xlsx",
    "model":"LightGBM fixed baseline, grouped by expID",
    "overall":overall,
    "dummy_baseline":dummy,
    "folds":folds,
    "coverage":coverage,
    "phosphate_descriptive_only":phosphate_summary,
    "guardrails":[
        "This test evaluates dataset/model credibility and phosphate coverage only.",
        "Descriptive Qe summaries are not moisture-swing rankings.",
        "No candidate is validated by this test.",
        "Grouped splits use expID to reduce leakage.",
        "ALCHEMI remains downstream until data/evidence gates pass."
    ],
    "next_decision":"If Gate A beats the naive baseline, proceed to uncertainty/OOD and exact phosphate gap mapping. Otherwise fix data/model validity before candidate selection."
}
(OUT/"msa-phosphate-gate-a-v0.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
print(json.dumps(report,indent=2,default=str))

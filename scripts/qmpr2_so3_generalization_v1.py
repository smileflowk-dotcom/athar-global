#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scale.candidate_factory_v1 import load_msa, CAT, NUM
from scale.batch_scorer_v1 import _model, FEATURES, FEATURE_CAT, FEATURE_NUM

OUT=ROOT/"03-validation"/"qmpr2-so3-generalization-v1.json"

TARGET_RESIN="QMPR-2"
IONS={
    "SO3^2-": (2.0, 7.20),
    "CO3^2-": (2.0, 10.33),
    "P2O7^4-": (4.0, 9.41),
}

def prep(df):
    x=df.copy()
    for c in FEATURE_CAT:
        x[c]=x[c].astype(str).fillna("unknown")
    for c in FEATURE_NUM:
        x[c]=pd.to_numeric(x[c],errors="coerce")
        med=x[c].median()
        x[c]=x[c].fillna(0.0 if pd.isna(med) else med)
    return x

df=load_msa().dropna(subset=["Qe","expID","resin"]).copy()
df["resin"]=df["resin"].astype(str)
target=df[df["resin"]==TARGET_RESIN].copy()
if target.empty:
    raise RuntimeError("QMPR-2 not found in public MSA dataset")

profile={}
for c in CAT:
    vals=target[c].dropna().astype(str)
    profile[c]=vals.mode().iloc[0] if len(vals) else "unknown"
for c in NUM:
    vals=pd.to_numeric(target[c],errors="coerce").dropna()
    profile[c]=float(vals.median()) if len(vals) else float(pd.to_numeric(df[c],errors="coerce").median())

hum=np.sort(pd.to_numeric(df["humidity"],errors="coerce").dropna().unique().astype(float))
low=float(np.quantile(hum,0.2))
high=float(np.quantile(hum,0.8))
if high<=low:
    low=float(hum[0]); high=float(hum[-1])

# Strict leave-one-resin-out training: QMPR-2 is excluded entirely.
train=prep(df[df["resin"]!=TARGET_RESIN].copy())
groups=np.array(sorted(train["expID"].dropna().unique()))
rng=np.random.default_rng(20261009)

models=[]
for i in range(12):
    sampled=rng.choice(groups,size=len(groups),replace=True)
    boot=pd.concat([train[train["expID"]==g] for g in sampled],ignore_index=True)
    m=_model(700+i)
    m.fit(boot[FEATURES],boot["Qe"].astype(float))
    models.append(m)

def candidate(ion, humidity):
    charge,pka=IONS[ion]
    r={**profile,"anion_charge":charge,"anion-pKa":pka,"humidity":humidity}
    for c in FEATURE_CAT:
        r[c]=str(r.get(c,"unknown"))
    for c in NUM+["anion_charge","anion-pKa","humidity"]:
        r[c]=float(r.get(c,0.0))
    return pd.DataFrame([r])

summary={}
for ion in IONS:
    lo=candidate(ion,low)
    hi=candidate(ion,high)
    pl=np.array([float(m.predict(lo[FEATURES])[0]) for m in models])
    ph=np.array([float(m.predict(hi[FEATURES])[0]) for m in models])
    swing=pl-ph
    summary[ion]={
        "humidity_low":low,
        "humidity_high":high,
        "pred_Qe_low_mean":float(pl.mean()),
        "pred_Qe_high_mean":float(ph.mean()),
        "pred_swing_mean":float(swing.mean()),
        "pred_swing_sd":float(swing.std(ddof=1)),
        "positive_fraction":float((swing>0).mean()),
        "ensemble_n":len(models),
    }

# Empirical QMPR-2 rows are used only as context, not for model fitting.
emp={
    "rows":int(len(target)),
    "expIDs":int(target["expID"].nunique()),
    "humidity_min":float(pd.to_numeric(target["humidity"],errors="coerce").min()),
    "humidity_max":float(pd.to_numeric(target["humidity"],errors="coerce").max()),
    "observed_anion_codes":sorted(pd.to_numeric(target.get("anion"),errors="coerce").dropna().astype(int).unique().tolist()) if "anion" in target else [],
}

s=summary["SO3^2-"]
# Conservative gate: signal positive in >=75% ensemble and magnitude exceeds its own SD.
if s["positive_fraction"]>=0.75 and s["pred_swing_mean"]>s["pred_swing_sd"]:
    status="PASS_ML_GENERALIZATION"
else:
    status="HOLD"

result={
    "status":status,
    "candidate":"QMPR-2 + SO3^2-",
    "test":"strict leave-one-resin-out ensemble generalization",
    "target_resin_excluded_from_training":True,
    "public_dataset":"PYU-pub/MSA-ML:data/MSA data.xlsx",
    "qmpr2_public_context":emp,
    "profile":profile,
    "summary":summary,
    "guardrails":[
        "SO3^2- is a virtual counterfactual represented by charge/pKa descriptors.",
        "QMPR-2 is fully excluded from training to test generalization.",
        "This test is ML evidence only, not atomistic or experimental validation.",
        "No atomistic claim is made until an exact or defensible QMPR-2 structural model is available.",
        "Literature/chemistry plausibility must be checked before expensive physics."
    ],
    "next_gate":"If PASS_ML_GENERALIZATION, resolve QMPR-2 chemistry/structure and run a defensible atomistic comparison against carbonate control; otherwise reject or hold."
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

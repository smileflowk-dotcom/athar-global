#!/usr/bin/env python3
from pathlib import Path
import sys,json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scale.candidate_factory_v1 import load_msa
from scale.batch_scorer_v1 import _model, FEATURES, FEATURE_CAT, FEATURE_NUM
from scale.candidate_factory_v2 import ION_MAP_V2

CANDS=[
 ("IRA900","C2O4^2-"),
 ("D201","citrate^3-"),
 ("QMPR-3","HCO3^-"),
 ("QMPR-2","B(OH)4^-"),
 ("QMPR-1","H2PO4^-"),
]
OUT=ROOT/"03-validation"/"autopilot-v1"
OUT.mkdir(parents=True,exist_ok=True)

df=load_msa().dropna(subset=["Qe","expID","resin"]).copy()
df["resin"]=df["resin"].astype(str)
hum=np.sort(pd.to_numeric(df["humidity"],errors="coerce").dropna().unique().astype(float))
low=float(np.quantile(hum,.2)); high=float(np.quantile(hum,.8))

def prep(x):
    x=x.copy()
    for c in FEATURE_CAT: x[c]=x[c].astype(str).fillna("unknown")
    for c in FEATURE_NUM:
        x[c]=pd.to_numeric(x[c],errors="coerce")
        med=x[c].median()
        x[c]=x[c].fillna(0.0 if pd.isna(med) else med)
    return x

def profile_for(resin):
    g=df[df.resin==resin]
    if g.empty: raise RuntimeError(f"missing resin {resin}")
    rec={}
    for c in FEATURE_CAT:
        vals=g[c].dropna().astype(str)
        rec[c]=vals.mode().iloc[0] if len(vals) else "unknown"
    for c in FEATURE_NUM:
        vals=pd.to_numeric(g[c],errors="coerce").dropna()
        rec[c]=float(vals.median()) if len(vals) else float(pd.to_numeric(df[c],errors="coerce").median())
    return rec

results=[]
for idx,(resin,ion) in enumerate(CANDS):
    train=prep(df[df.resin!=resin])
    groups=np.array(sorted(train.expID.dropna().unique()))
    rng=np.random.default_rng(20261010+idx)
    models=[]
    for i in range(12):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        boot=pd.concat([train[train.expID==g] for g in sampled],ignore_index=True)
        m=_model(9000+idx*20+i); m.fit(boot[FEATURES],boot.Qe.astype(float)); models.append(m)
    rec=profile_for(resin)
    charge,pka,_=ION_MAP_V2[ion]
    def row(h):
        r=dict(rec); r.update({"anion_charge":charge,"anion-pKa":pka,"humidity":h})
        return prep(pd.DataFrame([r]))
    lo=row(low); hi=row(high)
    swings=np.array([float(m.predict(lo[FEATURES])[0]-m.predict(hi[FEATURES])[0]) for m in models])
    mean=float(swings.mean()); sd=float(swings.std(ddof=1)); frac=float((swings>0).mean())
    passed=bool(frac>=.75 and mean>sd and mean>0.05)
    results.append({"resin":resin,"counterion":ion,"mean_swing":mean,"sd":sd,
                    "positive_fraction":frac,"pass":passed})

passers=[r for r in results if r["pass"]]
report={"status":"PASSERS_FOUND" if passers else "NO_PASSERS",
        "gate":"strict leave-one-resin-out ensemble",
        "results":results,"passers":passers,
        "guardrail":"ML generalization only; passers require two independent physics checks."}
(OUT/"generalization.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print("MATRIX="+json.dumps({"include":[{"candidate":r["resin"]+"__"+r["counterion"]} for r in passers]}))

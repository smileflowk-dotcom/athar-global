#!/usr/bin/env python3
from __future__ import annotations
import hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scale.candidate_factory_v1 import load_msa, CAT, NUM, _mode

# Exploration V2: keep known controls and add chemically plausible counter-ions.
# pKa values are descriptor approximations used only for ML screening; every novel ion
# remains a hypothesis until independent chemistry and experimental validation.
ION_MAP_V2={
  "CO3^2-": (2.0,10.33,"known_control"),
  "HCO3^-": (1.0,10.33,"new_exploration"),
  "PO4^3-": (3.0,12.32,"known"),
  "HPO4^2-": (2.0,7.19,"known"),
  "H2PO4^-": (1.0,7.21,"new_exploration"),
  "P2O7^4-": (4.0,9.41,"current_challenger"),
  "B(OH)4^-": (1.0,9.24,"new_exploration"),
  "C2O4^2-": (2.0,4.27,"new_exploration"),
  "citrate^3-": (3.0,6.40,"new_exploration"),
  "SO3^2-": (2.0,7.20,"new_exploration"),
}

def build_candidate_pool_v2(n=20000,seed=20261009):
    df=load_msa()
    work=df.dropna(subset=["resin","Qe","expID"]).copy()
    for c in CAT:
        if c in work: work[c]=work[c].astype(str)
    for c in NUM+["humidity"]:
        if c in work: work[c]=pd.to_numeric(work[c],errors="coerce")

    hum=np.sort(work["humidity"].dropna().unique().astype(float))
    lo=hum[hum<=np.quantile(hum,0.35)]
    hi=hum[hum>=np.quantile(hum,0.65)]
    if len(lo)==0: lo=hum[:1]
    if len(hi)==0: hi=hum[-1:]

    profiles={}
    for resin,g in work.groupby(work["resin"].astype(str)):
        rec={"resin":resin,"n_source_rows":int(len(g))}
        for c in CAT: rec[c]=str(_mode(g[c])) if c in g and _mode(g[c]) is not None else "unknown"
        for c in NUM:
            rec[c]=float(pd.to_numeric(g[c],errors="coerce").median()) if c in g else np.nan
        profiles[resin]=rec

    rng=np.random.default_rng(seed)
    resins=np.array(sorted(profiles))
    ions=list(ION_MAP_V2)
    combos=[(r,ion) for r in resins for ion in ions]
    rows=[]; k=0
    while len(rows)<n:
        resin,ion=combos[k%len(combos)]; k+=1
        g=work[work["resin"].astype(str)==resin]
        src=g.iloc[int(rng.integers(0,len(g)))]
        charge,pka,origin=ION_MAP_V2[ion]
        hlow=float(rng.choice(lo)); hhigh=float(rng.choice(hi))
        if hhigh<=hlow: hlow=float(hum[0]); hhigh=float(hum[-1])
        rec=dict(profiles[resin])
        for c in ["T","Cini","M/V"]:
            if c in src and pd.notna(src[c]): rec[c]=float(src[c])
        rec.update({
          "anion_code":ion,"counterion":ion,
          "anion_charge":charge,"anion-pKa":pka,"ion_origin":origin,
          "humidity_low":hlow,"humidity_high":hhigh,
          "source_expID":str(src["expID"]),
          "provenance_source_type":"virtual_counterfactual_exploration_v2",
          "provenance_source":"PYU-pub/MSA-ML:data/MSA data.xlsx",
          "stage":"20K","gate_status":"PENDING"
        })
        key="|".join([resin,ion,str(rec["source_expID"]),f"{hlow:.6g}",f"{hhigh:.6g}",str(len(rows))])
        rec["candidate_id"]="MSA2-"+hashlib.sha1(key.encode()).hexdigest()[:12]
        rows.append(rec)
    return pd.DataFrame(rows)

if __name__=="__main__":
    out=Path("scale/output-v2"); out.mkdir(parents=True,exist_ok=True)
    df=build_candidate_pool_v2()
    df.to_csv(out/"candidates-20k.csv",index=False)
    print({"rows":len(df),"resins":df.resin.nunique(),"ions":df.counterion.nunique()})

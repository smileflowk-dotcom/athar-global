#!/usr/bin/env python3
from __future__ import annotations
import hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

DATA_URL="https://raw.githubusercontent.com/PYU-pub/MSA-ML/main/data/MSA%20data.xlsx"
ION_MAP={
  2:("CO3^2-",2.0,10.33),
  3:("PO4^3-",3.0,12.32),
  5:("HPO4^2-",2.0,7.19),
  8:("P2O7^4-",4.0,9.41),
}
CAT=["base_type","pore","matrix","FG"]
NUM=["water_retention","BET","Vp","N%","FG:RU","IE-capacity","T","Cini","M/V"]

def load_msa(path="/tmp/MSA-data.xlsx"):
    p=Path(path)
    if not p.exists(): urllib.request.urlretrieve(DATA_URL,p)
    return pd.read_excel(p)

def _mode(s):
    s=s.dropna()
    return s.mode().iloc[0] if len(s) and len(s.mode()) else None

def build_candidate_pool(n=10000,seed=20261007):
    df=load_msa()
    work=df.dropna(subset=["resin","Qe","expID"]).copy()
    for c in CAT:
        if c in work: work[c]=work[c].astype(str)
    for c in NUM+["humidity"]:
        if c in work: work[c]=pd.to_numeric(work[c],errors="coerce")

    hum=np.sort(work["humidity"].dropna().unique().astype(float))
    if len(hum)<2: raise RuntimeError("Need at least two observed humidity values")
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
    rows=[]
    # Deterministic balanced coverage before random fill.
    combos=[(r,code) for r in resins for code in ION_MAP]
    k=0
    while len(rows)<n:
        resin,code=combos[k%len(combos)]
        k+=1
        g=work[work["resin"].astype(str)==resin]
        src=g.iloc[int(rng.integers(0,len(g)))]
        ion_name,charge,pka=ION_MAP[code]
        hlow=float(rng.choice(lo)); hhigh=float(rng.choice(hi))
        if hhigh<=hlow:
            hlow=float(hum[0]); hhigh=float(hum[-1])
        rec=dict(profiles[resin])
        # Reuse observed operating conditions wherever present.
        for c in ["T","Cini","M/V"]:
            if c in src and pd.notna(src[c]): rec[c]=float(src[c])
        rec.update({
          "anion_code":int(code),"counterion":ion_name,
          "anion_charge":charge,"anion-pKa":pka,
          "humidity_low":hlow,"humidity_high":hhigh,
          "source_expID":str(src["expID"]),
          "source_condition_anion":int(src["anion"]) if "anion" in src and pd.notna(src["anion"]) else None,
          "provenance_source_type":"virtual_counterfactual_from_public_MSA",
          "provenance_source":"PYU-pub/MSA-ML:data/MSA data.xlsx",
          "stage":"10K","gate_status":"PENDING"
        })
        key="|".join([resin,str(code),str(rec["source_expID"]),f"{hlow:.6g}",f"{hhigh:.6g}",f"{rec.get('T',np.nan):.6g}",f"{rec.get('Cini',np.nan):.6g}",f"{rec.get('M/V',np.nan):.6g}",str(len(rows))])
        rec["candidate_id"]="MSA-"+hashlib.sha1(key.encode()).hexdigest()[:12]
        rows.append(rec)
    return pd.DataFrame(rows)

if __name__=="__main__":
    out=Path("scale/output"); out.mkdir(parents=True,exist_ok=True)
    df=build_candidate_pool()
    df.to_csv(out/"candidates-10k.csv",index=False)
    print({"rows":len(df),"resins":df.resin.nunique(),"ions":df.counterion.nunique()})

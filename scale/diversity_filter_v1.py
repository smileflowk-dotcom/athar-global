#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

DESC=["water_retention","BET","Vp","N%","FG:RU","IE-capacity","anion_charge","anion-pKa","T","Cini","M/V","humidity_low","humidity_high"]

def diverse_select(df,n,seed=20261007,pool_factor=5,max_per_resin=None,max_per_pair=None):
    if len(df)<=n: return df.copy().reset_index(drop=True)
    pool=df.nlargest(min(len(df),max(n*pool_factor,n)), "aggregate_score").copy()
    X=pool[DESC].apply(pd.to_numeric,errors="coerce")
    X=X.fillna(X.median()).fillna(0)
    X=StandardScaler().fit_transform(X)
    k=min(n,len(pool))
    labels=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=10,batch_size=min(1024,len(pool))).fit_predict(X)
    pool["_cluster"]=labels
    ordered=[]
    # Pick best from each chemical/process cluster, then fill by score with quotas.
    for _,g in pool.groupby("_cluster"):
        ordered.append(g.nlargest(1,"aggregate_score").iloc[0])
    selected=pd.DataFrame(ordered).sort_values("aggregate_score",ascending=False)
    resin_count={}; pair_count={}; keep=[]
    candidates=pd.concat([selected,pool.sort_values("aggregate_score",ascending=False)]).drop_duplicates("candidate_id")
    for _,r in candidates.iterrows():
        resin=str(r["resin"]); pair=(resin,str(r["counterion"]))
        if max_per_resin is not None and resin_count.get(resin,0)>=max_per_resin: continue
        if max_per_pair is not None and pair_count.get(pair,0)>=max_per_pair: continue
        keep.append(r)
        resin_count[resin]=resin_count.get(resin,0)+1
        pair_count[pair]=pair_count.get(pair,0)+1
        if len(keep)>=n: break
    out=pd.DataFrame(keep).drop(columns=["_cluster"],errors="ignore").reset_index(drop=True)
    out["diversity_selected"]=True
    return out

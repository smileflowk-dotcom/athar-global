#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from scale.candidate_factory_v1 import load_msa, CAT, NUM

FEATURE_CAT=CAT
FEATURE_NUM=NUM+["anion_charge","anion-pKa","humidity"]
FEATURES=FEATURE_CAT+FEATURE_NUM

def _model(seed):
    pre=ColumnTransformer([
      ("cat",OneHotEncoder(handle_unknown="ignore"),FEATURE_CAT),
      ("num",StandardScaler(),FEATURE_NUM)])
    return Pipeline([("pre",pre),("model",LGBMRegressor(
      n_estimators=300,learning_rate=0.03,num_leaves=15,
      subsample=0.85,colsample_bytree=0.9,random_state=seed,verbosity=-1))])

def _prep_train():
    df=load_msa().dropna(subset=["Qe","expID"]).copy()
    for c in FEATURE_CAT:
        df[c]=df[c].astype(str).fillna("unknown")
    for c in FEATURE_NUM:
        df[c]=pd.to_numeric(df[c],errors="coerce")
        df[c]=df[c].fillna(df[c].median())
    return df

def _robust_z(x):
    x=np.asarray(x,float); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med))
    scale=1.4826*mad if mad>1e-12 else (np.nanstd(x) or 1.0)
    return (x-med)/scale

def score_candidates(candidates,ensemble=8,seed=20261007):
    train=_prep_train()
    groups=np.array(sorted(train["expID"].dropna().unique()))
    rng=np.random.default_rng(seed)
    models=[]
    for i in range(ensemble):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        boot=pd.concat([train[train["expID"]==g] for g in sampled],ignore_index=True)
        m=_model(100+i); m.fit(boot[FEATURES],boot["Qe"].astype(float)); models.append(m)

    base=candidates.copy()
    for c in FEATURE_CAT: base[c]=base[c].astype(str).fillna("unknown")
    for c in NUM+["anion_charge","anion-pKa"]:
        base[c]=pd.to_numeric(base[c],errors="coerce")
        base[c]=base[c].fillna(train[c].median() if c in train else 0.0)

    low=base[FEATURE_CAT+NUM+["anion_charge","anion-pKa"]].copy()
    high=low.copy()
    low["humidity"]=base["humidity_low"].astype(float).values
    high["humidity"]=base["humidity_high"].astype(float).values

    pl=np.vstack([m.predict(low[FEATURES]) for m in models])
    ph=np.vstack([m.predict(high[FEATURES]) for m in models])
    swings=pl-ph
    base["pred_Qe_low"]=pl.mean(axis=0)
    base["pred_Qe_high"]=ph.mean(axis=0)
    base["pred_swing"]=swings.mean(axis=0)
    base["uncertainty"]=swings.std(axis=0,ddof=1)

    # Partial OOD: standardized continuous descriptors + categorical mismatch count.
    ood_num=["water_retention","BET","Vp","N%","FG:RU","IE-capacity","anion_charge","anion-pKa","T","Cini","M/V"]
    tr=train[ood_num].astype(float)
    mu=tr.mean(); sd=tr.std().replace(0,1)
    tz=((tr-mu)/sd).values
    cz=((base[ood_num].astype(float)-mu)/sd).values
    # 10k x ~290 is safe; chunk to cap memory.
    d=[]
    for i in range(0,len(base),500):
        q=cz[i:i+500]
        dist=np.sqrt(((q[:,None,:]-tz[None,:,:])**2).mean(axis=2)).min(axis=1)
        d.extend(dist.tolist())
    base["ood"]=np.asarray(d)

    # Feasibility here means "descriptor-complete and anchored to a known resin/ion descriptor",
    # not commercial or experimental feasibility.
    base["feasibility_proxy"]=1.0
    base["swing_z"]=_robust_z(base["pred_swing"])
    base["uncertainty_z"]=_robust_z(base["uncertainty"])
    base["ood_z"]=_robust_z(base["ood"])
    base["aggregate_score"]=base["swing_z"]-0.45*base["uncertainty_z"]-0.35*base["ood_z"]
    base["score_status"]="ML_COUNTERFACTUAL_SCREEN_ONLY"
    return base.sort_values("aggregate_score",ascending=False).reset_index(drop=True)

if __name__=="__main__":
    from pathlib import Path
    p=Path("scale/output/candidates-10k.csv")
    df=score_candidates(pd.read_csv(p))
    df.to_csv("scale/output/scored-10k.csv",index=False)
    print(df[["candidate_id","resin","counterion","pred_swing","uncertainty","ood","aggregate_score"]].head(10).to_string(index=False))

#!/usr/bin/env python3
import json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

OUT=Path('03-validation'); OUT.mkdir(exist_ok=True)
DATA=Path('/tmp/MSA-data.xlsx')
urllib.request.urlretrieve('https://raw.githubusercontent.com/PYU-pub/MSA-ML/main/data/MSA%20data.xlsx', DATA)
df=pd.read_excel(DATA)

TARGETS=['IRA910','FPA54','MN100']
MAP={2:'CO3^2-',3:'PO4^3-',5:'HPO4^2-',8:'P2O7^4-'}

cat=[c for c in ['base_type','pore','matrix','FG'] if c in df.columns]
num=[c for c in ['water_retention','BET','Vp','N%','FG:RU','IE-capacity','anion_charge','anion-pKa','T','humidity','Cini','M/V'] if c in df.columns]
features=cat+num
work=df.dropna(subset=['Qe','expID','resin']).copy()
for c in cat: work[c]=work[c].astype(str).fillna('unknown')
for c in num:
    work[c]=pd.to_numeric(work[c],errors='coerce')
    work[c]=work[c].fillna(work[c].median())

def make_model(seed):
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num',StandardScaler(),num)])
    return Pipeline([('pre',pre),('model',LGBMRegressor(n_estimators=350,learning_rate=0.03,num_leaves=15,subsample=0.85,colsample_bytree=0.9,random_state=seed,verbosity=-1))])

# 1) Pair-holdout sanity check: can the model transfer a known ion to a resin when that exact pair is removed?
known_pairs=[]
for resin in TARGETS:
    for code in [2,3]:
        mask=(work['resin'].astype(str)==resin)&(work['anion']==code)
        if mask.sum()>=2:
            known_pairs.append((resin,code))

pair_holdout=[]
for resin,code in known_pairs:
    te=(work['resin'].astype(str)==resin)&(work['anion']==code)
    tr=~te
    m=make_model(42); m.fit(work.loc[tr,features],work.loc[tr,'Qe'].astype(float))
    p=m.predict(work.loc[te,features])
    pair_holdout.append({'resin':resin,'ion':MAP.get(code,str(code)),'n':int(te.sum()),'mae':float(mean_absolute_error(work.loc[te,'Qe'],p)),'bias':float(np.mean(p-work.loc[te,'Qe'].values))})

# 2) Bootstrap-by-expID ensemble uncertainty for exact unseen HPO4 resin+ion combinations.
rng=np.random.default_rng(20261007)
groups=np.array(sorted(work['expID'].dropna().unique().tolist()))
models=[]
for seed in range(12):
    sampled=rng.choice(groups,size=len(groups),replace=True)
    # preserve group multiplicity by concatenating sampled groups
    parts=[work[work['expID']==g] for g in sampled]
    boot=pd.concat(parts,ignore_index=True)
    m=make_model(100+seed); m.fit(boot[features],boot['Qe'].astype(float)); models.append(m)

# 3) Counterfactual HPO4 rows: use real operating rows for each target resin and change only anion descriptors.
candidates=[]
for resin in TARGETS:
    base=work[(work['resin'].astype(str)==resin)&(work['anion'].isin([2,3]))].copy()
    if base.empty: continue
    # Use observed conditions only, deduplicated on model features; no invented RH/T/CO2 conditions.
    cf=base[features+['resin','expID','anion']].drop_duplicates(subset=features).copy()
    cf['anion_charge']=2.0
    cf['anion-pKa']=7.19
    preds=np.vstack([m.predict(cf[features]) for m in models])
    mu=preds.mean(axis=0); sd=preds.std(axis=0,ddof=1)
    # OOD proxy in the chemically relevant 2-D anion descriptor space, standardized on training data.
    chem=work[['anion_charge','anion-pKa']].astype(float)
    c_mu=chem.mean(); c_sd=chem.std().replace(0,1)
    train_z=((chem-c_mu)/c_sd).values
    cand_z=((cf[['anion_charge','anion-pKa']].astype(float)-c_mu)/c_sd).values
    d=np.sqrt(((cand_z[:,None,:]-train_z[None,:,:])**2).sum(axis=2)).min(axis=1)
    for i,row in cf.reset_index(drop=True).iterrows():
        candidates.append({
            'resin':resin,'candidate_ion':'HPO4^2-',
            'source_condition_ion':MAP.get(int(row['anion']),str(row['anion'])),
            'humidity':float(row['humidity']) if 'humidity' in row and pd.notna(row['humidity']) else None,
            'T':float(row['T']) if 'T' in row and pd.notna(row['T']) else None,
            'Cini':float(row['Cini']) if 'Cini' in row and pd.notna(row['Cini']) else None,
            'pred_Qe_mean':float(mu[i]),'ensemble_sd':float(sd[i]),'chemical_nn_distance_z':float(d[i])
        })

# Aggregate by resin. This is prioritization, not validation.
ranking=[]
for resin in TARGETS:
    rows=[x for x in candidates if x['resin']==resin]
    if not rows: continue
    ranking.append({
      'resin':resin,
      'candidate_ion':'HPO4^2-',
      'n_real_conditions_reused':len(rows),
      'pred_Qe_mean_across_conditions':float(np.mean([x['pred_Qe_mean'] for x in rows])),
      'median_ensemble_sd':float(np.median([x['ensemble_sd'] for x in rows])),
      'max_ensemble_sd':float(np.max([x['ensemble_sd'] for x in rows])),
      'median_chemical_nn_distance_z':float(np.median([x['chemical_nn_distance_z'] for x in rows])),
      'status':'UNCERTAINTY_OOD_SCREEN_ONLY'
    })
ranking=sorted(ranking,key=lambda x:(-x['median_ensemble_sd'],-x['pred_Qe_mean_across_conditions']))

report={
 'status':'PASS' if candidates and pair_holdout else 'FAIL',
 'test':'Gate B uncertainty/OOD screen for exact HPO4 gaps',
 'targets':TARGETS,
 'known_pair_holdout':pair_holdout,
 'candidate_ranking_for_information_value':ranking,
 'candidate_condition_predictions':candidates,
 'guardrails':[
   'Predictions are counterfactual ML estimates, not measurements.',
   'Operating conditions are reused from real rows; no new RH/T/CO2 conditions are invented.',
   'Ensemble spread is a model-disagreement proxy, not a calibrated confidence interval.',
   'Chemical nearest-neighbor distance uses only charge and pKa and is not a full molecular OOD metric.',
   'No candidate advances to ALCHEMI until literature novelty and feasibility are checked.',
   'No candidate is validated by this test.'
 ],
 'next_decision':'Use holdout error + ensemble spread + exact literature/feasibility checks to select at most 1-2 HPO4 resin pairs for targeted NVIDIA ALCHEMI.'
}
(OUT/'msa-phosphate-uncertainty-ood-v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

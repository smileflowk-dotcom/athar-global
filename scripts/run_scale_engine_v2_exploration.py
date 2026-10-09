#!/usr/bin/env python3
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scale.candidate_factory_v2 import build_candidate_pool_v2
from scale.batch_scorer_v1 import score_candidates
from scale.gates_v1 import run_gates

OUT=ROOT/"scale"/"output-v2"; OUT.mkdir(parents=True,exist_ok=True)

print("V2 STEP 1 Candidate pool 20K")
cand=build_candidate_pool_v2(n=20000)
cand.to_csv(OUT/"candidates-20k.csv",index=False)

print("V2 STEP 2 ML counterfactual scoring")
scored=score_candidates(cand,ensemble=8,seed=20261009)
scored.to_csv(OUT/"scored-20k.csv",index=False)

print("V2 STEP 3 staged diversity gates")
stages=run_gates(scored,OUT)

# Explicitly summarize novel challengers separately from known controls.
g20=stages["20"].copy()
novel=g20[g20["ion_origin"]=="new_exploration"].sort_values("aggregate_score",ascending=False)
top_novel=novel.head(10)
top_novel.to_csv(OUT/"top-novel-challengers.csv",index=False)

summary={
 "scientific_status":"HYPOTHESIS_SCREENING_ONLY_NOT_VALIDATED",
 "rows_screened":len(cand),
 "resins":int(cand.resin.nunique()),
 "counterions":sorted(cand.counterion.unique().tolist()),
 "new_counterions":sorted(cand.loc[cand.ion_origin=="new_exploration","counterion"].unique().tolist()),
 "top_novel_challengers":[
   {
    "candidate_id":r.candidate_id,
    "resin":r.resin,
    "counterion":r.counterion,
    "pred_swing":float(r.pred_swing),
    "uncertainty":float(r.uncertainty),
    "ood":float(r.ood),
    "aggregate_score":float(r.aggregate_score)
   } for _,r in top_novel.iterrows()
 ],
 "guardrails":[
   "Novel ions are virtual counterfactuals represented by charge/pKa descriptors.",
   "ML ranking is hypothesis generation, not physical validation.",
   "High OOD or uncertainty candidates require stronger skepticism.",
   "No candidate is promoted without chemistry review and atomistic/independent validation."
 ]
}
(OUT/"discovery-v2-summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

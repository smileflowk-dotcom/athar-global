from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scale.candidate_factory_v2 import build_candidate_pool_v2
from scale.batch_scorer_v1 import score_candidates

OUT=ROOT/"scale"/"output-v3"
OUT.mkdir(parents=True,exist_ok=True)
cand=build_candidate_pool_v2(n=30000,seed=20261010)
sc=score_candidates(cand,ensemble=12,seed=20261010)

blocked={"SO3^2-","P2O7^4-","CO3^2-","PO4^3-","HPO4^2-"}
q=sc[~sc.counterion.isin(blocked)].copy()
q=q[q.uncertainty<=q.uncertainty.quantile(.60)]
q=q[q.ood<=q.ood.quantile(.60)]

agg=q.groupby(["resin","counterion"],as_index=False).agg(
 pred_swing=("pred_swing","median"),
 uncertainty=("uncertainty","median"),
 ood=("ood","median"),
 aggregate_score=("aggregate_score","median"),
 n_configs=("candidate_id","count"))
agg=agg.sort_values("aggregate_score",ascending=False)

picked=[]
used_r=set()
used_i=set()
for _,r in agg.iterrows():
    if r.resin in used_r or r.counterion in used_i:
        continue
    picked.append(r)
    used_r.add(r.resin)
    used_i.add(r.counterion)
    if len(picked)==5:
        break

top=__import__("pandas").DataFrame(picked)
top.to_csv(OUT/"top-5.csv",index=False)
summary={"status":"HYPOTHESIS_SCREENING_ONLY_NOT_VALIDATED","screened":len(cand),
"excluded":sorted(blocked),"top_5":top.to_dict("records"),
"next_gate":"chemistry review then leave-one-resin-out generalization"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))

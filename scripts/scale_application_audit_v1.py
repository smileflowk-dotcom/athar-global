#!/usr/bin/env python3
from pathlib import Path
import sys, json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scale.candidate_factory_v1 import load_msa

OUT=ROOT/"scale"/"output"
# Prefer the new overall ranking; fall back to legacy top-5 gate.
p=OUT/"ranking-overall-top5.csv"
top=pd.read_csv(p if p.exists() else OUT/"gate-5.csv")
msa=load_msa()

ION_CODE={"CO3^2-":2,"PO4^3-":3,"HPO4^2-":5,"P2O7^4-":8}
rows=[]
for _,r in top.iterrows():
    resin=str(r["resin"]); ion=str(r["counterion"]); code=ION_CODE.get(ion)
    exact=((msa["resin"].astype(str)==resin)&(msa["anion"]==code)) if code is not None else pd.Series(False,index=msa.index)
    rec=r.to_dict()
    rec["observed_exact_pair_in_MSA"]=bool(exact.any())
    rec["observed_rows_exact_pair"]=int(exact.sum())
    rec["coverage_role"]="KNOWN_IN_MSA" if exact.any() else "DATA_GAP_IN_MSA"
    rec["next_action"]="EVIDENCE_FEASIBILITY_AND_PHYSICS_AS_NEEDED"
    rows.append(rec)

aud=pd.DataFrame(rows).sort_values("aggregate_score",ascending=False).reset_index(drop=True)
aud.to_csv(OUT/"gate-5-application-audit.csv",index=False)

known=aud[aud["coverage_role"]=="KNOWN_IN_MSA"]
gaps=aud[aud["coverage_role"]=="DATA_GAP_IN_MSA"]

report={
 "status":"PASS",
 "test":"Application audit of best overall candidates",
 "top5_total":int(len(aud)),
 "known_in_msa":int(len(known)),
 "data_gap_in_msa":int(len(gaps)),
 "best_overall":aud[["candidate_id","resin","counterion","aggregate_score","pred_swing","uncertainty","ood","coverage_role","observed_rows_exact_pair","next_action"]].to_dict(orient="records"),
 "decision":"Keep both known and data-gap candidates. MSA/literature novelty is not an elimination gate; it is metadata for interpretation and a secondary ranking.",
 "guardrails":[
   "Known candidates can still be among the best experimental choices.",
   "Absence from MSA does not prove novelty in the literature.",
   "No candidate advances to a discovery claim without independent evidence and experiment."
 ]
}
(OUT/"gate-5-application-audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))

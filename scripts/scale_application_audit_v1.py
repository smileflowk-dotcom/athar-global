#!/usr/bin/env python3
from pathlib import Path
import sys
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scale.candidate_factory_v1 import load_msa

OUT=ROOT/"scale"/"output"
top=pd.read_csv(OUT/"gate-5.csv")
msa=load_msa()

ION_CODE={"CO3^2-":2,"PO4^3-":3,"HPO4^2-":5,"P2O7^4-":8}

rows=[]
for _,r in top.iterrows():
    resin=str(r["resin"]); ion=str(r["counterion"]); code=ION_CODE.get(ion)
    exact=((msa["resin"].astype(str)==resin)&(msa["anion"]==code)) if code is not None else pd.Series(False,index=msa.index)
    known=bool(exact.any())
    rec=r.to_dict()
    rec["observed_exact_pair_in_MSA"]=known
    rec["observed_rows_exact_pair"]=int(exact.sum())
    rec["application_role"]="KNOWN_CONTROL" if known else "DATA_GAP_CANDIDATE"
    rec["next_action"]="CONTROL_REFERENCE" if known else "EVIDENCE_NOVELTY_FEASIBILITY"
    rows.append(rec)

aud=pd.DataFrame(rows)
aud.to_csv(OUT/"gate-5-application-audit.csv",index=False)

gaps=aud[aud["application_role"]=="DATA_GAP_CANDIDATE"].sort_values("aggregate_score",ascending=False)
controls=aud[aud["application_role"]=="KNOWN_CONTROL"].sort_values("aggregate_score",ascending=False)

report={
 "status":"PASS",
 "test":"Apply Scale Engine V1 top-5 to exact-pair coverage audit before expensive physics",
 "top5_total":int(len(aud)),
 "data_gap_candidates":int(len(gaps)),
 "known_controls":int(len(controls)),
 "data_gap":gaps[["candidate_id","resin","counterion","aggregate_score","observed_rows_exact_pair","next_action"]].to_dict(orient="records"),
 "controls":controls[["candidate_id","resin","counterion","aggregate_score","observed_rows_exact_pair","next_action"]].to_dict(orient="records"),
 "decision":"Do not spend ALCHEMI on every top-5 row. Known exact pairs are controls; only data gaps advance to novelty/feasibility evidence before physics.",
 "guardrails":[
   "Absence from MSA does not prove literature novelty.",
   "Known in MSA does not mean experimentally superior.",
   "This audit prevents computational ranking from being confused with discovery ranking."
 ]
}
(OUT/"gate-5-application-audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))

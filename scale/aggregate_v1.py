#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scale.candidate_factory_v1 import load_msa

ION_CODE={"CO3^2-":2,"PO4^3-":3,"HPO4^2-":5,"P2O7^4-":8}

def _label_coverage(df):
    msa=load_msa()
    out=df.copy()
    observed=[]; nrows=[]
    for _,r in out.iterrows():
        resin=str(r["resin"]); ion=str(r["counterion"]); code=ION_CODE.get(ion)
        mask=((msa["resin"].astype(str)==resin)&(msa["anion"]==code)) if code is not None else pd.Series(False,index=msa.index)
        observed.append(bool(mask.any()))
        nrows.append(int(mask.sum()))
    out["observed_exact_pair_in_MSA"]=observed
    out["observed_rows_exact_pair"]=nrows
    out["novelty_bucket"]=out["observed_exact_pair_in_MSA"].map({True:"KNOWN_IN_MSA",False:"DATA_GAP_IN_MSA"})
    return out

def _records(df,n=5):
    cols=["candidate_id","resin","counterion","pred_swing","uncertainty","ood","aggregate_score",
          "humidity_low","humidity_high","observed_exact_pair_in_MSA","observed_rows_exact_pair",
          "novelty_bucket","gate_status"]
    return df.head(n)[cols].to_dict(orient="records")

def write_report(stages,outdir="scale/output"):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)

    # Primary objective: best candidates overall, whether known or not.
    top20=_label_coverage(stages["20"].sort_values("aggregate_score",ascending=False).copy())
    overall=top20.head(5).copy()

    # Secondary objective: novelty/data-gap exploration. This never replaces the overall ranking.
    gaps=top20[~top20["observed_exact_pair_in_MSA"]].sort_values("aggregate_score",ascending=False).head(5).copy()

    overall.to_csv(out/"ranking-overall-top5.csv",index=False)
    gaps.to_csv(out/"ranking-data-gap-top5.csv",index=False)

    summary={
      "status":"PASS_COMPUTATIONAL_PIPELINE",
      "counts":{k:int(len(v)) for k,v in stages.items()},
      "primary_objective":"BEST_CANDIDATES_OVERALL_FOR_EXPERIMENTAL_TESTING",
      "best_candidates_overall":_records(overall,5),
      "data_gap_candidates":_records(gaps,5) if len(gaps) else [],
      "novelty_policy":"Novelty/data-gap status is metadata and a secondary ranking signal, never an elimination gate.",
      "scientific_status":"SCREENING_ONLY_NOT_VALIDATED",
      "guardrails":[
        "Known literature or MSA candidates remain eligible if they are experimentally promising.",
        "Absence from MSA does not prove literature novelty.",
        "10K entries are virtual resin x counter-ion x operating-condition configurations, not 10K experimentally known materials.",
        "ML outputs are counterfactual predictions from the public MSA dataset, not measurements.",
        "OOD is a partial descriptor-space proxy, not full chemical-domain validation.",
        "Final promotion still requires evidence, feasibility, targeted physics where useful, independent cross-check, and lab testing.",
        "No candidate is a discovery or lab-validated result."
      ]
    }
    (out/"scale-ranking-v2.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    lines=["# Scale Engine V2 — Candidate Ranking","",
           "**Primary objective:** identify the best candidates to test, whether already known or not.","",
           "## Funnel","- 10K generated","- 1K ML/OOD/uncertainty + diversity","- 100 computational shortlist","- 20 pre-physics shortlist","- dual ranking: overall + data-gap","",
           "## Best candidates overall"]
    for i,r in enumerate(summary["best_candidates_overall"],1):
        lines.append(f"{i}. **{r['resin']} + {r['counterion']}** — predicted swing {r['pred_swing']:.4f}, uncertainty {r['uncertainty']:.4f}, OOD {r['ood']:.3f}, score {r['aggregate_score']:.3f}, {r['novelty_bucket']}")
    lines+=["","## Data-gap candidates (secondary ranking)"]
    if summary["data_gap_candidates"]:
        for i,r in enumerate(summary["data_gap_candidates"],1):
            lines.append(f"{i}. **{r['resin']} + {r['counterion']}** — score {r['aggregate_score']:.3f}")
    else:
        lines.append("- None in the current top-20.")
    lines+=["","## Guardrails"]+[f"- {x}" for x in summary["guardrails"]]
    (out/"scale-ranking-v2.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary

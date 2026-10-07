#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

def write_report(stages,outdir="scale/output"):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    top=stages["5"].copy()
    cols=["candidate_id","resin","counterion","pred_swing","uncertainty","ood","aggregate_score","humidity_low","humidity_high","gate_status"]
    ranking=top[cols].to_dict(orient="records")
    summary={
      "status":"PASS_COMPUTATIONAL_PIPELINE",
      "counts":{k:int(len(v)) for k,v in stages.items()},
      "top5":ranking,
      "scientific_status":"SCREENING_ONLY_NOT_VALIDATED",
      "guardrails":[
        "10K entries are virtual resin x counter-ion x operating-condition configurations, not 10K experimentally known materials.",
        "ML outputs are counterfactual predictions from the public MSA dataset, not measurements.",
        "OOD is a partial descriptor-space proxy, not full chemical-domain validation.",
        "Automatic 100->20->5 promotion is computational pre-ranking only; literature novelty, real feasibility, ALCHEMI and an independent cross-check are still required.",
        "No candidate is a discovery or lab-validated result."
      ]
    }
    (out/"scale-ranking-v1.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Scale Engine V1 — Computational Ranking","",
           "**Status:** screening only; not a material validation.","",
           "## Funnel",f"- 10K generated","- 1K ML/OOD/uncertainty + diversity","- 100 computational shortlist","- 20 pre-physics shortlist","- 5 finalists awaiting evidence + ALCHEMI + independent cross-check","",
           "## Top 5"]
    for i,r in enumerate(ranking,1):
        lines.append(f"{i}. **{r['resin']} + {r['counterion']}** — predicted swing {r['pred_swing']:.4f}, uncertainty {r['uncertainty']:.4f}, OOD {r['ood']:.3f}, score {r['aggregate_score']:.3f}")
    lines+=["","## Guardrails"]+[f"- {x}" for x in summary["guardrails"]]
    (out/"scale-ranking-v1.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary

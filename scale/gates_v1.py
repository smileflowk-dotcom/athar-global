#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
from scale.diversity_filter_v1 import diverse_select

def run_gates(scored,outdir="scale/output"):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    # Gate 10K -> 1K: quality plus broad diversity.
    g1=diverse_select(scored,1000,pool_factor=4,max_per_resin=100,max_per_pair=40)
    g1["stage"]="1K"; g1["gate_status"]="PASS"
    g1.to_csv(out/"gate-1k.csv",index=False)

    # 1K -> 100: tighter diversity; evidence/novelty remains a downstream requirement.
    g100=diverse_select(g1,100,pool_factor=6,max_per_resin=12,max_per_pair=5)
    g100["stage"]="100"; g100["gate_status"]="PASS_COMPUTATIONAL"
    g100["evidence_gate"]="REQUIRED"
    g100.to_csv(out/"gate-100.csv",index=False)

    # 100 -> 20: computational pre-shortlist only, not a novelty or feasibility validation.
    g20=diverse_select(g100,20,pool_factor=5,max_per_resin=4,max_per_pair=2)
    g20["stage"]="20"; g20["gate_status"]="HOLD_FOR_EVIDENCE_AND_HEAVIER_CHECKS"
    g20.to_csv(out/"gate-20.csv",index=False)

    # 20 -> 5: screening finalists. No ALCHEMI result is fabricated here.
    g5=diverse_select(g20,5,pool_factor=4,max_per_resin=2,max_per_pair=1)
    g5["stage"]="5"; g5["gate_status"]="HOLD_FOR_ALCHEMI_AND_INDEPENDENT_CROSSCHECK"
    g5.to_csv(out/"gate-5.csv",index=False)
    return {"1K":g1,"100":g100,"20":g20,"5":g5}

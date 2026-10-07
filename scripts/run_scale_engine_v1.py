#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scale.candidate_factory_v1 import build_candidate_pool
from scale.batch_scorer_v1 import score_candidates
from scale.gates_v1 import run_gates
from scale.aggregate_v1 import write_report

OUT=ROOT/"scale"/"output"; OUT.mkdir(parents=True,exist_ok=True)
print("STEP 1/5 Candidate Factory V1")
cand=build_candidate_pool(n=10000)
cand.to_csv(OUT/"candidates-10k.csv",index=False)
print("generated",len(cand))

print("STEP 2/5 Batch scorer 10K -> ranked")
scored=score_candidates(cand,ensemble=8)
scored.to_csv(OUT/"scored-10k.csv",index=False)

print("STEP 3/5 Diversity filter + STEP 4/5 automatic gates")
stages=run_gates(scored,OUT)

print("STEP 5/5 Aggregator/ranking")
report=write_report(stages,OUT)
print(report)

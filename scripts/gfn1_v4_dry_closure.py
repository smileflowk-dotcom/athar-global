#!/usr/bin/env python3
"""Finish ONLY missing dry-state independent GFN1 evidence; never promote a candidate.

Previously completed run 38175588297 supplies matching PO4 and CO3 3/6/9
hydration raw states. This script runs only 0 H2O using the same V4 adapter
and a continuation of the same FIRE optimizer (120 + at most 120 steps).
Acceptance remains finite energies/forces and fmax <= 0.40 eV/Angstrom.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemistry.v4.physics import make_source

OUT = Path("03-validation/gfn1-dry-closure")
PREVIOUS_RUN = 38175588297
PREVIOUS_ARTIFACT = 11716871740
SOURCE_ANCHOR = '\nrunner=run_aimnet if method=="aimnet" else run_gfn1'
OLD = (
    "        FIRE(a,logfile=None,dt=.05,maxstep=.10).run(fmax=.30,steps=120)\n"
    "        e=float(a.get_potential_energy())\n"
    "        fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
    '        return {"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40),"e":e,"fmax":fm}'
)
NEW = (
    "        opt=FIRE(a,logfile=None,dt=.05,maxstep=.10)\n"
    "        first_converged=bool(opt.run(fmax=.30,steps=120))\n"
    "        e=float(a.get_potential_energy())\n"
    "        fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
    '        first={"fmax":fm,"energy":e,"optimizer_converged":first_converged,"steps":int(opt.nsteps)}\n'
    "        retried=False\n"
    "        final_converged=first_converged\n"
    "        if not (np.isfinite(e) and np.isfinite(fm) and fm<=.40):\n"
    "            retried=True\n"
    "            final_converged=bool(opt.run(fmax=.30,steps=120))\n"
    "            e=float(a.get_potential_energy())\n"
    "            fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
    '        ok=bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40)\n'
    '        return {"ok":ok,"e":e,"fmax":fm,"first":first,"retried":retried,'
    '"steps":int(opt.nsteps),"optimizer_converged":final_converged,'
    '"reason":"accepted" if ok else "nonfinite_or_force_above_original_limit"}'
)
RUNNER = r'''
import json
from pathlib import Path
out=Path(os.environ["V4_DRY_RAW"])
out.parent.mkdir(parents=True,exist_ok=True)
states=[]
pairs=[]
for testion in ["CO3^2-","PO4^3-"]:
    for trial in (0,1):
        pair={}
        for label in ("near","far"):
            item=run_gfn1(testion,0,trial,label)
            pair[label]=item
            states.append({"ion":testion,"n_h2o":0,"trial":trial,"state":label,**item})
            out.write_text(json.dumps({"states":states,"rows":pairs,
                "status":"PARTIAL_DIAGNOSTIC_ONLY","source":"same V4 adapter; 120+120 FIRE"},indent=2)+"\n")
            print("DRY_STATE",testion,trial,label,
                  "ok",item.get("ok"),"fmax",item.get("fmax"),
                  "steps",item.get("steps"),flush=True)
        valid=bool(pair["near"].get("ok") and pair["far"].get("ok"))
        pairs.append({"ion":testion,"n_h2o":0,"trial":trial,
                      "ok":valid,
                      "delta_eV":pair["near"]["e"]-pair["far"]["e"] if valid else None,
                      "detail":pair})
        out.write_text(json.dumps({"states":states,"rows":pairs,
            "status":"DRY_DIAGNOSTIC_ONLY","source":"same V4 adapter; 120+120 FIRE"},indent=2)+"\n")
print("DRY_GFN1_DONE",len(pairs),flush=True)
'''

def instrument():
    source, _, _ = make_source("IRA900__PO4^3-", "gfn1")
    if source.count(SOURCE_ANCHOR) != 1 or source.count(OLD) != 1:
        raise RuntimeError("V4 source/force preflight changed: STOP")
    if source.count("max_iterations=1200") != 1:
        raise RuntimeError("GFN1 calculator settings changed: STOP")
    source = source.split(SOURCE_ANCHOR, 1)[0]
    source = source.replace(OLD, NEW)
    result = source + "\n" + RUNNER
    compile(result, "v4-dry-independent-gfn1.py", "exec")
    return result

def valid(row):
    try:
        a,b=(row["detail"][s] for s in ("near","far"))
        return bool(row["ok"] and a["ok"] and b["ok"] and
                    all(math.isfinite(float(x["fmax"])) and
                        float(x["fmax"]) <= 0.40 and
                        math.isfinite(float(x["e"])) for x in (a,b)) and
                    math.isfinite(float(row["delta_eV"])) and
                    abs((float(a["e"])-float(b["e"]))-float(row["delta_eV"])) < 1e-5)
    except (KeyError,TypeError,ValueError,OverflowError):
        return False

def finish():
    old_dir=OUT/"reused"
    old=json.loads((old_dir/"raw-rows.json").read_text())
    diag=json.loads((old_dir/"diagnostic.json").read_text())
    dry=json.loads((OUT/"dry-rows.json").read_text())
    if (diag.get("scientifically_confirmed") is not False or
        diag.get("eligible_for_v4_cache_or_promotion") is not False or
        diag.get("hydratations_tested") != [3,6,9] or
        diag.get("trials") != [0,1] or not diag.get("matched_carbonate_control") or
        "fmax <= 0.40" not in diag.get("original_acceptance_preserved","")):
        raise RuntimeError("Historical diagnostic provenance invalid: HOLD")
    pinned={"numpy":"2.5.4","ase":"3.29.0","rdkit":"2026.9.1","tblite":"0.7.0"}
    old_deps=(old_dir/"dependencies.txt").read_text()
    for package,version in pinned.items():
        if package+"=="+version not in old_deps.splitlines():
            raise RuntimeError("Old critical GFN1 dependency changed: HOLD")
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Runtime critical GFN1 dependency drift: HOLD")
    rows=list(old["rows"])+list(dry["rows"])
    if len(old["rows"]) != 12 or len(dry["rows"]) != 4 or len(dry.get("states",[])) != 8:
        raise RuntimeError("Incomplete paired raw states: HOLD")
    expected={(ion,n,trial) for ion in ("PO4^3-","CO3^2-")
              for n in (0,3,6,9) for trial in (0,1)}
    observed=[(r["ion"],r["n_h2o"],r["trial"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Unmatched/duplicate hydration seeds: HOLD")
    summary={}
    for ion in ("PO4^3-","CO3^2-"):
        count,med={},{}
        for n in (0,3,6,9):
            pairs=[r for r in rows if r["ion"]==ion and r["n_h2o"]==n]
            good=[float(r["delta_eV"]) for r in pairs if valid(r)]
            count[str(n)]=len(good)
            med[str(n)]=statistics.median(good) if len(good)>=2 else None
        shift=(med["9"]-med["0"] if med["0"] is not None and med["9"] is not None else None)
        summary[ion]={"accepted_pairs":count,"median_delta_eV":med,
                      "shift_eV":shift,"gate_pairs_all_2":all(x>=2 for x in count.values())}
    target,control=summary["PO4^3-"],summary["CO3^2-"]
    gates_ok=(target["gate_pairs_all_2"] and control["gate_pairs_all_2"]
              and target["shift_eV"] is not None and target["shift_eV"]>0)
    provenance={}
    for p in (old_dir/"diagnostic.json",old_dir/"raw-rows.json",
              old_dir/"dependencies.txt",OUT/"dry-rows.json",OUT/"generated-source.py"):
        provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report={"candidate":"IRA900__PO4^3-","method":"independent GFN1-xTB",
            "status":"READY_FOR_SCIENTIFIC_REVIEW" if gates_ok else "HOLD",
            "scientifically_confirmed":False,"eligible_for_v4_cache_or_promotion":False,
            "lab_validated":False,"old_run":PREVIOUS_RUN,"old_artifact_id":PREVIOUS_ARTIFACT,
            "new_work":"ONLY 0 H2O, target + matched carbonate, seeds 0/1, near/far",
            "reused_work":"3/6/9 H2O paired GFN1 states + carbonate from prior run",
            "gfn1_original_force_limit_eV_A":0.40,"fire_target_eV_A":0.30,
            "fire_budget":"up to 120+120 iterations, same optimizer, no threshold relaxation",
            "summary":summary,"computational_gates_met":bool(gates_ok),
            "source_hashes":provenance,
            "warning":"Local fragment microhydration energy only; not real sorbent adsorption, RH, free energy or experimental proof."}
    (OUT/"merged-raw-rows.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
    (OUT/"decision.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"decision.md").write_text(
        "# Targeted independent GFN1 closure (NO LAB CONFIRMATION)\n\n"+
        f"Status: **{report['status']}**. Reused run {PREVIOUS_RUN} (3/6/9 plus controls);"+
        " computed only 0 H2O target and carbonate.\n\n"+
        f"PO4 accepted pairs: {target['accepted_pairs']}; shift eV: {target['shift_eV']}.\n\n"+
        f"CO3 accepted pairs: {control['accepted_pairs']}; shift eV: {control['shift_eV']}.\n\n"+
        "Independent computational screen only. No automatic scientific promotion, scaling or lab assertion.\n")
    print("GFN1_DRY_CLOSURE",json.dumps({"status":report["status"],
          "computational_gates_met":gates_ok,"target":target,"control":control}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight",action="store_true")
    actions.add_argument("--run",action="store_true")
    actions.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    if args.finalize:
        finish()
        return
    source=instrument()
    if args.preflight:
        print("V4_DRY_PREFLIGHT_OK: unchanged fmax<=.40, FIRE=.30, continuation 120+120; no promotion")
        return
    OUT.mkdir(parents=True,exist_ok=True)
    script=OUT/"generated-source.py"
    script.write_text(source,encoding="utf-8")
    env={**os.environ,"CANDIDATE":"IRA900__PO4^3-","METHOD":"gfn1",
         "V4_DRY_RAW":str(OUT/"dry-rows.json")}
    with (OUT/"solver-output.log").open("w",encoding="utf-8") as logs:
        proc=subprocess.run([sys.executable,str(script)],env=env,stdout=logs,
                            stderr=subprocess.STDOUT,check=False)
    if proc.returncode:
        raise RuntimeError(f"Independent GFN1 dry-only run failed ({proc.returncode}); see preserved logs")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Evidence-only replay of IRA900 PO4 GFN1 failures at 3/6/9 H2O.

Uses exactly the existing V4 GFN1 adapter and force acceptance rule.
Records per-state e/fmax, optimizer exit, errors and pair outcomes. It
does NOT produce a V4 physics PASS, overwrite history, or change gates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemistry.v4.physics import make_source

CANDIDATE = "IRA900__PO4^3-"
OUTPUT = ROOT / "chemistry/v4/diagnostics/gfn1-po4-hydrated-trial0.json"
RUN_LOOP = r'''
from pathlib import Path
import json, os
trials = [int(t) for t in os.environ["V4_DIAG_TRIALS"].split(",")]
out_path = Path(os.environ["V4_DIAG_OUT"])
out_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
def reason(v):
    if v.get("error"):
        return "CALCULATOR_OR_OPTIMIZER_EXCEPTION"
    if not v.get("ok"):
        if v.get("fmax") is not None and v["fmax"] > .40:
            return "FINAL_FORCE_ABOVE_EXISTING_0_40_LIMIT"
        return "NONFINITE_OR_OTHER_REJECTION"
    if not v.get("optimizer_converged", False):
        return "ACCEPTED_BY_OLD_0_40_GATE_BUT_NOT_0_30_OPTIMIZER_TARGET"
    return "ACCEPTED"

for nw in [3, 6, 9]:
    for trial in trials:
        states = {}
        for state in ("near", "far"):
            print("DIAGNOSTIC_CASE", nw, trial, state, flush=True)
            detail = run_gfn1("PO4^3-", nw, trial, state)
            detail["diagnostic_reason"] = reason(detail)
            states[state] = detail
            rows.append({"n_h2o": nw, "trial": trial, "state": state, **detail})
            out_path.write_text(json.dumps({
                "status": "DIAGNOSTIC_ONLY_NOT_A_VALIDATION",
                "candidate": "IRA900__PO4^3-", "method": "gfn1",
                "historical_run": 38095586192,
                "hydration_states": [3, 6, 9],
                "trials": trials,
                "force_acceptance_eV_per_A": .40,
                "optimizer_target_eV_per_A": .30,
                "optimizer_steps": 120,
                "rows": rows,
                "scientific_decision": "HOLD",
                "note": "No matched carbonate recalc, no 0/9 shift, no independent PASS."
            }, indent=2) + "\n")
        valid = bool(states["near"].get("ok") and states["far"].get("ok"))
        delta = states["near"]["e"] - states["far"]["e"] if valid else None
        print("DIAGNOSTIC_PAIR", nw, trial, "valid", valid, "delta_eV", delta, flush=True)
'''
def diagnostic_source():
    source, _, _ = make_source(CANDIDATE, "gfn1")
    anchor = "\nrunner=run_aimnet if method==\"aimnet\" else run_gfn1"
    if source.count(anchor) != 1:
        raise RuntimeError("Unexpected upstream runner boundary")
    source = source.split(anchor, 1)[0]
    old = "FIRE(a,logfile=None,dt=.05,maxstep=.10).run(fmax=.30,steps=120)"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected GFN1 optimizer source")
    source = source.replace(old, "optimizer_converged = " + old)
    old_ret = 'return {"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40),"e":e,"fmax":fm}'
    if source.count(old_ret) != 1:
        raise RuntimeError("Existing force threshold changed: STOP")
    source = source.replace(old_ret, old_ret.replace(
        '{"ok":', '{"optimizer_converged":bool(optimizer_converged),"ok":'))
    result = source + "\n" + RUN_LOOP
    compile(result, "<v4-gfn1-diagnostics>", "exec")
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", default="0", help="Comma-separated trial IDs from 0,1,2")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    chosen = [x.strip() for x in args.trials.split(",")]
    if not chosen or len(set(chosen)) != len(chosen) or any(x not in {"0", "1", "2"} for x in chosen):
        p.error("Only unique trials 0,1,2 are allowed")
    source = diagnostic_source()
    if args.self_test:
        print("GFN1_DIAGNOSTIC_PREFLIGHT_OK; original thresholds intact")
        return
    output = OUTPUT.with_name("gfn1-po4-hydrated-trials-" + "-".join(chosen) + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CANDIDATE=CANDIDATE, METHOD="gfn1",
               V4_DIAG_TRIALS=",".join(chosen), V4_DIAG_OUT=str(output))
    with tempfile.TemporaryDirectory() as d:
        executable = Path(d) / "gfn1-v4-diagnostic-runtime.py"
        executable.write_text(source, encoding="utf-8")
        log = output.with_suffix(".log")
        with log.open("w", encoding="utf-8") as raw:
            exit_code = subprocess.call([sys.executable, str(executable)],
                                        cwd=ROOT, env=env, stdout=raw,
                                        stderr=subprocess.STDOUT)
    if exit_code:
        print("GFN1_DIAGNOSTIC_ERROR", exit_code, "log_tail:",
              log.read_text(errors="replace")[-1300:])
        raise SystemExit(exit_code)
    data = json.loads(output.read_text(encoding="utf-8"))
    for n in [3, 6, 9]:
        subset = [x for x in data["rows"] if x["n_h2o"] == n]
        failures = [x for x in subset if not x.get("ok")]
        print("GFN1_DIAGNOSTIC", n, "H2O", len(subset),
              "states", len(failures), "rejections",
              [{"trial": x["trial"], "state": x["state"],
                "reason": x["diagnostic_reason"], "fmax": x.get("fmax"),
                "error": x.get("error")} for x in failures])
    print("DIAGNOSTIC_ONLY", output)

if __name__ == "__main__":
    main()

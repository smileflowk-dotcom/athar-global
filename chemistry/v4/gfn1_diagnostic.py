#!/usr/bin/env python3
"""Bounded GFN1 diagnosis for IRA900__PO4^3-; no promotion or relaxed gates.

Generates the reviewed V4 GFN1 source, preserves its method, geometries, seeds,
charge and accepted-force threshold, and ONLY extends FIRE's iteration budget.
Runs target at 3/6/9 H2O; the historical dry and carbonate results are NOT
reused as matched proof of a changed optimization protocol.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from chemistry.v4.physics import make_source

CANDIDATE = "IRA900__PO4^3-"
OUTPUT = Path("chemistry/v4/diagnostics/gfn1-po4-3-6-9")
HYDRATIONS = ("3", "6", "9")


def once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Upstream GFN1 adapter drift: expected one, got {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def diagnostic_source() -> tuple[str, Path]:
    src, original_report, key = make_source(CANDIDATE, "gfn1")
    if key != "PO4^3-":
        raise RuntimeError("Wrong target ion")
    if src.count("for nw in [0,3,6,9]:") != 2:
        raise RuntimeError("Expected one sweep and one summary hydration loop")
    # The target/control loop is present in both execution and summary blocks.
    old_ions = 'for testion in ["CO3^2-",ion]:'
    if src.count(old_ions) != 2:
        raise RuntimeError("Upstream target/control source shape changed")
    src = src.replace(old_ions, 'for testion in [ion]:')
    src = once(src,
        'for nw in [0,3,6,9]:\n    for trial in [0,1,2]:',
        'for nw in [3,6,9]:\n    for trial in [0,1,2]:')
    src = once(src,
        'for nw in [0,3,6,9]:\n        vals=',
        'for nw in [3,6,9]:\n        vals=')
    # More optimizer steps, *not* a looser fmax convergence or acceptance gate.
    src = once(src,
        '        FIRE(a,logfile=None,dt=.05,maxstep=.10).run(fmax=.30,steps=120)',
        '        opt=FIRE(a,logfile=None,dt=.05,maxstep=.10)\n'
        '        converged=bool(opt.run(fmax=.30,steps=240))')
    src = once(src,
        '        return {"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40),"e":e,"fmax":fm}',
        '        finite=bool(np.isfinite(e) and np.isfinite(fm))\n'
        '        ok=bool(finite and fm<=.40)\n'
        '        return {"ok":ok,"e":e if finite else None,"fmax":fm if finite else None,'
        '"optimizer_converged":converged,"optimizer_steps":int(opt.nsteps),'
        '"failure_reason":None if ok else ("NONFINITE" if not finite else "FMAX_GT_0.40")}')
    src = once(src,
        '        return {"ok":False,"error":str(ex)}',
        '        return {"ok":False,"error":str(ex),"failure_reason":"EXCEPTION"}')
    src = once(src,
        '        pair[state]=runner(testion,nw,trial,state)',
        '        pair[state]=runner(testion,nw,trial,state)\n'
        '        print("GFN1_DIAGNOSTIC_STATE",json.dumps({"ion":testion,"hydration":nw,'
        '"trial":trial,"state":state,**pair[state]}),flush=True)')
    src = once(src,
        '      rows.append({"ion":testion,"n_h2o":nw,"trial":trial,"ok":bool(ok),"delta_eV":delta,"detail":pair})',
        '      rows.append({"ion":testion,"n_h2o":nw,"trial":trial,"ok":bool(ok),"delta_eV":delta,"detail":pair})\n'
        '      diag=Path("chemistry/v4/diagnostics/gfn1-po4-3-6-9")\n'
        '      diag.mkdir(parents=True,exist_ok=True)\n'
        '      (diag/"partial-rows.json").write_text(json.dumps(rows,indent=2)+"\\n")\n'
        '      print("GFN1_DIAGNOSTIC_PAIR",json.dumps({"hydration":nw,"trial":trial,"ok":bool(ok),'
        '"near":pair["near"],"far":pair["far"]}),flush=True)')
    src = once(src,
        '    shift=med["9"]-med["0"] if med["0"] is not None and med["9"] is not None else None',
        '    shift=None  # diagnostic excludes dry point; cannot compute or claim a swing')
    src = once(src,
        'status="PASS" if s["positive"] is True and s["pairs"]["0"]>=1 and s["pairs"]["9"]>=1 else ("HOLD" if s["positive"] is False else "PARTIAL")',
        'status="DIAGNOSTIC_ONLY"')
    # Keep raw method report in separate diagnostics path; avoid V4 PASS cache.
    src = once(src,
        'p=Path("03-validation/autopilot-v1"); p.mkdir(parents=True,exist_ok=True)',
        'p=Path("chemistry/v4/diagnostics/gfn1-po4-3-6-9"); p.mkdir(parents=True,exist_ok=True)')
    compile(src, "generated_gfn1_diagnostic.py", "exec")
    return src, original_report


def assess(raw: dict) -> dict:
    rows = raw["rows"]
    tally = {}
    for h in (3, 6, 9):
        subset = [r for r in rows if r["n_h2o"] == h]
        if len(subset) != 3 or len({r["trial"] for r in subset}) != 3:
            raise RuntimeError(f"Incomplete triplicate at {h} H2O")
        accepted = sum(r["ok"] for r in subset)
        failures = [
            {"trial": r["trial"], "near": r["detail"]["near"],
             "far": r["detail"]["far"]}
            for r in subset if not r["ok"]
        ]
        tally[str(h)] = {"valid_pairs": accepted, "failures": failures}
    enough = all(x["valid_pairs"] >= 2 for x in tally.values())
    return {
        "candidate": CANDIDATE, "method": "GFN1-xTB",
        "status": "DIAGNOSTIC_ONLY", "readiness_for_matched_full_recheck": enough,
        "hydration_points": [3, 6, 9], "trials": 3, "pair_gate_unchanged": 2,
        "force_acceptance_gate_eV_A_unchanged": 0.40,
        "fire_fmax_eV_A_unchanged": 0.30, "fire_steps_old": 120,
        "fire_steps_diagnostic": 240,
        "tally": tally,
        "notes": ["No carbonate control or dry-state rerun in this diagnostic",
                  "No hydration shift, independent confirmation, material discovery or lab claim",
                  "If diagnostic succeeds, rerun target AND matched carbonate with the same reviewed protocol"],
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    src, _ = diagnostic_source()
    generated = OUTPUT / "generated-source.py"
    generated.write_text(src, encoding="utf-8")
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    print("GFN1_DIAGNOSTIC_SOURCE_SHA256", sha, flush=True)
    # The underlying script writes a full raw report including per-state
    # energy, fmax, convergence and exception information; never overwrites
    # original historical artifacts or the canonical V4 cache.
    with tempfile.TemporaryDirectory() as tmp:
        entry = Path(tmp) / "run.py"
        entry.write_text(src, encoding="utf-8")
        result = subprocess.run([sys.executable, str(entry)], check=False)
    if result.returncode:
        (OUTPUT / "execution-failure.json").write_text(json.dumps(
            {"returncode": result.returncode, "source_sha256": sha}, indent=2) + "\n")
        return result.returncode
    path = OUTPUT / "IRA900__PO43-__gfn1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    outcome = assess(raw)
    outcome["source_sha256"] = sha
    (OUTPUT / "diagnosis.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print("GFN1_DIAGNOSTIC_RESULT", json.dumps({
        "status": outcome["status"], "tally": {
            h: x["valid_pairs"] for h, x in outcome["tally"].items()},
        "ready_for_matched_recheck": outcome["readiness_for_matched_full_recheck"]
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

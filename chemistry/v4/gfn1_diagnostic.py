#!/usr/bin/env python3
"""Bounded IRA900 phosphate GFN1 diagnostic, not a physics promotion run.

Reuse the existing V4 GFN1 adapter and its original acceptance criteria.
Test only 3/6/9 H2O, two fixed geometry seeds, plus matched carbonate.
Continue the SAME optimizer if a configuration fails the original force gate;
keep both initial and retried force data. Never label this as full robustness.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from physics import make_source

CANDIDATE = "IRA900__PO4^3-"
HYDRATIONS = (3, 6, 9)
TRIALS = (0, 1)
ROOT = Path("03-validation/gfn1-targeted")
OUT = Path("chemistry/v4/output")


def replace_exact(source, old, new, count=1):
    occurrences = source.count(old)
    if occurrences != count:
        raise RuntimeError(f"Adapter preflight failed: {old!r} occurs {occurrences} times, expected {count}")
    return source.replace(old, new)


def instrument_existing_source(candidate):
    source, report, _ = make_source(candidate, "gfn1")
    # Restrict work to the previously failing hydration regimes, not a new discovery scan.
    source = replace_exact(source, "for nw in [0,3,6,9]:", "for nw in [3,6,9]:", 2)
    source = replace_exact(source, "for trial in [0,1,2]:", "for trial in [0,1]:")
    source = replace_exact(
        source,
        'shift=med["9"]-med["0"] if med["0"] is not None and med["9"] is not None else None',
        'shift=med["9"]-med["0"] if med.get("0") is not None and med.get("9") is not None else None',
    )
    source = replace_exact(source, 's["pairs"]["0"]>=1', 's["pairs"].get("0",0)>=1')
    old = (
        "        FIRE(a,logfile=None,dt=.05,maxstep=.10).run(fmax=.30,steps=120)\n"
        "        e=float(a.get_potential_energy())\n"
        "        fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
        '        return {"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40),"e":e,"fmax":fm}'
    )
    new = (
        "        opt=FIRE(a,logfile=None,dt=.05,maxstep=.10)\n"
        "        initial_converged=bool(opt.run(fmax=.30,steps=120))\n"
        "        e=float(a.get_potential_energy())\n"
        "        fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
        '        initial={"e":e,"fmax":fm,"optimizer_converged":initial_converged,"steps":int(opt.nsteps)}\n'
        "        retried=False\n"
        "        final_converged=initial_converged\n"
        "        if not (np.isfinite(e) and np.isfinite(fm) and fm<=.40):\n"
        "            retried=True\n"
        "            final_converged=bool(opt.run(fmax=.30,steps=120))\n"
        "            e=float(a.get_potential_energy())\n"
        "            fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))\n"
        "        accepted=bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40)\n"
        '        reason="accepted" if accepted else "nonfinite_or_force_above_original_limit"\n'
        '        return {"ok":accepted,"e":e,"fmax":fm,"first":initial,'
        '"retried":retried,"steps":int(opt.nsteps),'
        '"optimizer_converged":final_converged,"reason":reason}'
    )
    source = replace_exact(source, old, new)
    source = replace_exact(
        source, 'return {"ok":False,"error":str(ex)}',
        'return {"ok":False,"error":str(ex),"reason":"calculator_or_optimizer_exception"}'
    )
    compile(source, "gfn1_targeted_instrumented.py", "exec")
    return source, report


def classify(raw):
    states = []
    pairs = []
    for row in raw["rows"]:
        detail = row["detail"]
        state_info = []
        for state in ("near", "far"):
            info = detail[state]
            fail_reason = info.get("reason", "accepted" if info.get("ok") else "unknown")
            state_info.append({
                "state": state, "accepted": bool(info.get("ok")),
                "initial_fmax": info.get("first", {}).get("fmax"),
                "final_fmax": info.get("fmax"),
                "initial_optimizer_converged": info.get("first", {}).get("optimizer_converged"),
                "final_optimizer_converged": info.get("optimizer_converged"),
                "retry_used": bool(info.get("retried")),
                "steps": info.get("steps"),
                "reason": fail_reason,
                "error": info.get("error"),
            })
        states.append({"ion": row["ion"], "hydration": row["n_h2o"],
                       "trial": row["trial"], "near_far": state_info})
        pairs.append({"ion": row["ion"], "hydration": row["n_h2o"],
                      "trial": row["trial"], "accepted": bool(row["ok"]),
                      "delta_eV": row["delta_eV"]})
    count = {}
    for ion in ("CO3^2-", "PO4^3-"):
        count[ion] = {str(h): sum(1 for p in pairs
                                  if p["ion"] == ion and p["hydration"] == h and p["accepted"])
                      for h in HYDRATIONS}
    return states, pairs, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source, report_path = instrument_existing_source(CANDIDATE)
    if args.preflight:
        print("GFN1_TARGETED_PREFLIGHT_OK: 3/6/9 H2O, 2 trials, matched carbonate, same force threshold")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        generated = Path(temp) / "instrumented_gfn1.py"
        generated.write_text(source, encoding="utf-8")
        log = ROOT / "solver-output.log"
        env = os.environ.copy()
        env.update({"CANDIDATE": CANDIDATE, "METHOD": "gfn1"})
        with log.open("w", encoding="utf-8") as stream:
            execution = subprocess.run([sys.executable, str(generated)],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False)
    if execution.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"GFN1 probe failed (exit {execution.returncode}); preserve solver-output.log")
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if len(raw.get("rows", [])) != 2 * len(HYDRATIONS) * len(TRIALS):
        raise RuntimeError("Probe missing geometries; cannot use result")
    states, pairs, counts = classify(raw)
    import shutil
    shutil.copyfile(report_path, ROOT / "raw-rows.json")
    result = {
        "candidate": CANDIDATE, "method": "gfn1",
        "status": "DIAGNOSTIC_ONLY_HOLD",
        "scientifically_confirmed": False,
        "eligible_for_v4_cache_or_promotion": False,
        "hydrate_0_missing": True,
        "hydratations_tested": list(HYDRATIONS),
        "trials": list(TRIALS),
        "matched_carbonate_control": True,
        "original_acceptance_preserved": "finite energies/forces and fmax <= 0.40 eV/Angstrom",
        "optimizer_target_preserved": "fmax = 0.30 eV/Angstrom",
        "optimizer_retry": "continue same deterministic FIRE optimization up to 120 additional steps if first pass is not accepted",
        "valid_pairs": counts,
        "pairs": pairs, "states": states,
        "note": "Local-site microhydration only; 0 H2O not repeated; diagnostic cannot promote to PASS.",
    }
    (ROOT / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# IRA900 PO4 GFN1 targeted diagnostic (NOT PHYSICS PASS)",
        "",
        "Only 3/6/9 H2O, 2 deterministic geometries, matched carbonate.",
        "No threshold change, no million scale, no lab validation, not full 4-point robustness.",
        "",
        "| Ion | 3 H2O | 6 H2O | 9 H2O |",
        "| --- | --- | --- | --- |",
    ]
    for ion in ("CO3^2-", "PO4^3-"):
        lines.append(f"| {ion} | " + " | ".join(f"{counts[ion][str(h)]}/2" for h in HYDRATIONS) + " |")
    lines.append("")
    lines.append("See diagnostic.json for pre/post-retry forces and per-state errors; raw-rows.json and solver-output.log retained.")
    (ROOT / "diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("GFN1_TARGETED_DIAGNOSTIC", json.dumps({"status": "HOLD", "valid_pairs": counts,
          "raw_trace": str(ROOT / "diagnostic.json")}), flush=True)


if __name__ == "__main__":
    main()

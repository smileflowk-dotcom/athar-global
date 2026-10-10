#!/usr/bin/env python3
"""Compatibility runner: reuse the proven V1 physics implementation,
but tighten GFN1 convergence before accepting any NEW V4 evidence.

Fast 0/9 H2O check is only a PRECHECK. Full robustness is a separate gate.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path("scripts/autopilot_physics_v1.py")
OLD = '"ok":bool(np.isfinite(e) and np.isfinite(fm))'
NEW = '"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--method", required=True, choices=["gfn1", "aimnet"])
    args = parser.parse_args()
    if "__" not in args.candidate:
        parser.error("candidate must be resin__counterion")
    text = SOURCE.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise SystemExit("Upstream GFN1 acceptance rule changed: refusing unreviewed patch")
    text = text.replace(OLD, NEW)
    if text.count("max_iterations=800") == 1:
        text = text.replace("max_iterations=800", "max_iterations=1200")
    if text.count("fmax=.35,steps=90") == 1:
        text = text.replace("fmax=.35,steps=90", "fmax=.30,steps=120")
    with tempfile.TemporaryDirectory() as tmp:
        child = Path(tmp) / "atomistic_test.py"
        child.write_text(text, encoding="utf-8")
        env = os.environ.copy()
        env.update({"CANDIDATE": args.candidate, "METHOD": args.method})
        result = subprocess.run([sys.executable, str(child)], env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    # The original script writes JSON to 03-validation/autopilot-v1.
    # Action uploads it into chemistry/v4/cache/new, separately from old files.
    print("V4_PRECHECK_ONLY", args.candidate, args.method, flush=True)


if __name__ == "__main__":
    main()

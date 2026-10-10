#!/usr/bin/env python3
"""Targeted V4 full physics confirmation using existing AIMNet2 and GFN1 code.

Four water states x three geometries, matched carbonate control, near/far CO2.
All results remain computational local-site proxies, not real-world proof.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ION_SMILES = {
    "C2O4^2-": ("[O-]C(=O)C(=O)[O-]", -2),
    "citrate^3-": ("O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]", -3),
    "HCO3^-": ("O=C(O)[O-]", -1),
    "B(OH)4^-": ("[B-](O)(O)(O)O", -1),
    "H2PO4^-": ("OP(=O)(O)[O-]", -1),
    "P2O7^4-": ("[O-]P(=O)([O-])OP(=O)([O-])[O-]", -4),
    "SO3^2-": ("[O-]S(=O)[O-]", -2),
}
SITES = {
    "IRA900": "C[N+](C)(C)Cc1ccccc1",
    "D201": "C[N+](C)(C)Cc1ccccc1",
    "QMPR-1": "C[N+](C)(C)CC(=O)c1ccccc1",
    "QMPR-2": "C[N+](C)(C)CC(=O)c1ccccc1",
    "QMPR-3": "C[N+](C)(C)CC(=O)c1ccccc1",
}
HYDRATIONS = ("0", "3", "6", "9")


def make_source(candidate, method):
    resin, ion = candidate.split("__", 1)
    if resin not in SITES or ion not in ION_SMILES:
        raise ValueError("No reviewed model mapping for this resin/ion pair")
    if method == "aimnet":
        text = Path("scripts/alchemi_ira900_p2o7_robustness_v1.py").read_text()
        ion_smi, ion_charge = ION_SMILES[ion]
        new_ions = ("IONS={\n"
                    "  \"carbonate\":{\"smiles\":\"[O-]C(=O)[O-]\",\"charge\":-2},\n"
                    f"  \"target\":{{\"smiles\":{json.dumps(ion_smi)},\"charge\":{ion_charge}}},\n"
                    "}\n")
        text, n = re.subn(r"IONS=\{.*?\}\n(?=HYDRATION=)", lambda _: new_ions, text, flags=re.S)
        if n != 1:
            raise RuntimeError("Robustness source changed (ION model)")
        text, n = re.subn(r'SITE_SMILES="[^"]+"', lambda _: "SITE_SMILES=" + json.dumps(SITES[resin]), text)
        if n != 1:
            raise RuntimeError("Robustness source changed (site)")
        old = "    if n==2:\n        u=np.array([[1,0,0],[-1,0,0]],float)"
        new = ("    if n==1:\n"
               "        u=np.array([[1,0,0]],float)\n"
               "    elif n==3:\n"
               "        u=np.array([[1,0,0],[-0.5,0.8660254,0],[-0.5,-0.8660254,0]],float)\n"
               "    elif n==2:\n"
               "        u=np.array([[1,0,0],[-1,0,0]],float)")
        if text.count(old) != 1:
            raise RuntimeError("Robustness source changed (site vector mapping)")
        text = text.replace(old, new)
        src_report = Path("03-validation/alchemi-ira900-p2o7-robustness-v1.json")
        return text, src_report, "target"

    text = Path("scripts/autopilot_physics_v1.py").read_text()
    if text.count(' "H2PO4^-":("OP(=O)(O)[O-]",-1),') != 1:
        raise RuntimeError("Autopilot V1 source changed (ion mapping)")
    extra = (' "P2O7^4-":("[O-]P(=O)([O-])OP(=O)([O-])[O-]",-4),\n'
             ' "SO3^2-":("[O-]S(=O)[O-]",-2),\n')
    text = text.replace(' "H2PO4^-":("OP(=O)(O)[O-]",-1),',
                        ' "H2PO4^-":("OP(=O)(O)[O-]",-1),\n' + extra.rstrip("\n"))
    old = '"ok":bool(np.isfinite(e) and np.isfinite(fm))'
    if text.count(old) != 1 or text.count("for nw in [0,9]:") != 2:
        raise RuntimeError("Autopilot V1 source changed (convergence or hydration)")
    text = text.replace(old, '"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40)')
    text = text.replace("for nw in [0,9]:", "for nw in [0,3,6,9]:")
    text = text.replace("for trial in [0,1]:", "for trial in [0,1,2]:")
    text = text.replace("max_iterations=800", "max_iterations=1200")
    text = text.replace("fmax=.35,steps=90", "fmax=.30,steps=120")
    src_report = (Path("03-validation/autopilot-v1") /
                  (candidate.replace("/", "_").replace("^", "")
                   .replace("(", "").replace(")", "") + "__gfn1.json"))
    return text, src_report, ion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--method", choices=["gfn1", "aimnet"], required=True)
    args = p.parse_args()
    text, source_report, ion_key = make_source(args.candidate, args.method)
    with tempfile.TemporaryDirectory() as temp:
        script = Path(temp) / "existing_model_with_v4_adapter.py"
        script.write_text(text, encoding="utf-8")
        env = os.environ.copy()
        env.update({"CANDIDATE": args.candidate, "METHOD": args.method})
        proc = subprocess.run([sys.executable, str(script)], env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    report = json.loads(source_report.read_text(encoding="utf-8"))
    summary = report["summary"][ion_key]
    if args.method == "aimnet":
        counts = [summary[h]["valid_pairs"] for h in (
            "dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o")]
        shift = summary["hydration_shift_0_to_9_eV"]
        physical_gate = report["status"] == "PASS"
    else:
        counts = [summary["pairs"].get(h, 0) for h in HYDRATIONS]
        shift = summary["shift_eV"]
        physical_gate = report["status"] == "PASS"
    passed = bool(physical_gate and shift is not None and shift > 0 and min(counts) >= 2)
    output = {
        "candidate": args.candidate,
        "method": args.method,
        "status": "PASS" if passed else "HOLD",
        "full_robust": True, "hydration_points": [0, 3, 6, 9],
        "trials": 3, "valid_pairs": counts,
        "hydration_shift_eV": shift,
        "source_script": "existing V1 physics + V4 convergence / hydration adapter",
        "physics_proxy_only": True,
        "note": "Local-site microhydration proxy; no lab validation or measured adsorption",
    }
    path = Path("chemistry/v4/cache/new/v4-robust-result.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("V4_FULL_ROBUSTNESS", json.dumps(output), flush=True)


if __name__ == "__main__":
    main()

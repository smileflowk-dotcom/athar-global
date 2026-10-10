#!/usr/bin/env python3
"""Chemistry Discovery Engine V4 -- initial adapter: CO2 moisture-swing.
Streams virtual configurations; rigorous tests only on shortlisted chemistry.
A screen is NOT a material discovery or experimental validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scale.candidate_factory_v1 import load_msa
from scale.candidate_factory_v2 import ION_MAP_V2
from scale.batch_scorer_v1 import _model, FEATURES, FEATURE_CAT, FEATURE_NUM, NUM

SUPPORTED_RESINS = {"IRA900", "D201", "QMPR-1", "QMPR-2", "QMPR-3"}
SUPPORTED_IONS = {"CO3^2-", "C2O4^2-", "citrate^3-", "HCO3^-", "B(OH)4^-", "H2PO4^-"}
ANCHORS = [("QMPR-1", "H2PO4^-"), ("QMPR-3", "HCO3^-"), ("IRA900", "C2O4^2-")]
GUARDRAIL = ("Virtual operating configurations are not unique material discoveries. "
             "Local-site binding-energy proxies are not adsorption free energies, "
             "laboratory measurements, or proof of full-polymer performance.")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def prep(df, reference):
    x = df.copy()
    for col in FEATURE_CAT:
        x[col] = x[col].fillna("unknown").astype(str)
    for col in FEATURE_NUM:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        fallback = pd.to_numeric(reference[col], errors="coerce").median()
        x[col] = x[col].fillna(0.0 if pd.isna(fallback) else fallback)
    return x


def profiles(df):
    """Create 1 representative measured-resin descriptor record per resin."""
    out = {}
    for resin, group in df.dropna(subset=["resin"]).groupby(df["resin"].astype(str)):
        row = {"resin": str(resin)}
        for name in FEATURE_CAT:
            s = group[name].dropna().astype(str)
            row[name] = s.mode().iloc[0] if len(s) else "unknown"
        for name in NUM:
            s = pd.to_numeric(group[name], errors="coerce").dropna()
            row[name] = float(s.median()) if len(s) else np.nan
        out[str(resin)] = row
    return out


def model_data():
    data = load_msa().dropna(subset=["Qe", "expID", "resin"]).copy()
    for col in FEATURE_CAT:
        data[col] = data[col].fillna("unknown").astype(str)
    return prep(data, data)


def discover(args):
    from scale.candidate_factory_v1 import _mode
    df = model_data()
    model = _model(20261010)
    model.fit(df[FEATURES], df["Qe"].astype(float))
    records = profiles(df)
    pairs = [(r, i) for r in sorted(records) for i in ION_MAP_V2]
    if not pairs:
        raise RuntimeError("No traceable MSA resin/ion templates")
    rng = np.random.default_rng(args.seed)
    humidity = pd.to_numeric(df["humidity"], errors="coerce").dropna().to_numpy()
    if len(humidity) < 2:
        raise RuntimeError("Need measured humidity observations")
    hlow = np.sort(humidity[humidity <= np.quantile(humidity, 0.35)])
    hhigh = np.sort(humidity[humidity >= np.quantile(humidity, 0.65)])
    if not len(hlow) or not len(hhigh) or max(hhigh) <= min(hlow):
        raise RuntimeError("Humidity range invalid")

    # Precompute observed operating-condition pools for each resin. No synthetic
    # polymer structures or untraceable chemistry are invented.
    condition_pools = {}
    for r in records:
        sub = df.loc[df["resin"].astype(str) == r]
        condition_pools[r] = sub[["T", "Cini", "M/V"]].to_numpy(dtype=float)

    # Fast OOD proxy uses experimental feature bounds; the slow nearest-neighbor
    # OOD and ensemble uncertainty are reserved for survivors only.
    obs = df[["anion_charge", "anion-pKa", "T", "Cini", "M/V"]].astype(float)
    lower, upper = obs.quantile(0.01), obs.quantile(0.99)
    span = (upper - lower).replace(0, 1.0)

    best = {}
    screened = 0
    while screened < args.configurations:
        count = min(args.chunk, args.configurations - screened)
        pair_ids = (np.arange(screened, screened + count) + args.seed) % len(pairs)
        base = pd.DataFrame([records[pairs[p][0]].copy() for p in pair_ids])
        resins = [pairs[p][0] for p in pair_ids]
        ions = [pairs[p][1] for p in pair_ids]
        base["anion_charge"] = [ION_MAP_V2[i][0] for i in ions]
        base["anion-pKa"] = [ION_MAP_V2[i][1] for i in ions]

        # Sample conditions only from observations of the corresponding resin.
        for r in set(resins):
            ix = np.flatnonzero(np.asarray(resins) == r)
            pool = condition_pools[r]
            chosen = pool[rng.integers(0, len(pool), size=len(ix))]
            base.loc[ix, ["T", "Cini", "M/V"]] = chosen
        lo = rng.choice(hlow, size=count)
        hi = rng.choice(hhigh, size=count)
        bad = hi <= lo
        lo[bad], hi[bad] = np.min(humidity), np.max(humidity)
        low = prep(base.assign(humidity=lo), df)
        high = prep(base.assign(humidity=hi), df)
        pred_swing = model.predict(low[FEATURES]) - model.predict(high[FEATURES])
        numeric = base[lower.index].astype(float)
        ood = (np.maximum(0, lower - numeric) + np.maximum(0, numeric - upper)).div(span).sum(axis=1).to_numpy()
        score = pred_swing - 0.20 * ood
        # Keep strongest traceable configuration per chemical pair, not only
        # the globally top-scoring operating conditions.
        frame = pd.DataFrame({
            "resin": resins, "counterion": ions, "pred_swing_coarse": pred_swing,
            "ood_proxy": ood, "coarse_score": score,
            "humidity_low": lo, "humidity_high": hi,
            "source": "PYU-pub/MSA-ML:data/MSA data.xlsx",
        })
        for (r, i), group in frame.groupby(["resin", "counterion"], sort=False):
            item = group.loc[group["coarse_score"].idxmax()].to_dict()
            old = best.get((r, i))
            if old is None or item["coarse_score"] > old["coarse_score"]:
                best[(r, i)] = item
        screened += count
        print(f"COARSE_STREAM_EVALUATED {screened}/{args.configurations}", flush=True)

    available = sorted(best.values(), key=lambda x: x["coarse_score"], reverse=True)
    picked, counts_resin, counts_ion = [], {}, {}
    for row in available:
        r, i = row["resin"], row["counterion"]
        if r not in SUPPORTED_RESINS or i not in SUPPORTED_IONS:
            continue
        if counts_resin.get(r, 0) >= 2 or counts_ion.get(i, 0) >= 2:
            continue
        picked.append({**row, "selection": "score_diversity"})
        counts_resin[r] = counts_resin.get(r, 0) + 1
        counts_ion[i] = counts_ion.get(i, 0) + 1
        if len(picked) == args.shortlist:
            break
    # Retain previously investigated strong hypotheses: no restart of discovery.
    selected_pairs = {(x["resin"], x["counterion"]) for x in picked}
    for pair in ANCHORS:
        if pair in best and pair not in selected_pairs:
            picked.append({**best[pair], "selection": "prior_evidence_anchor"})
    result = {
        "domain": "co2_moisture_swing", "engine": "chemistry-discovery-v4",
        "status": "COARSE_HYPOTHESIS_SCREEN_ONLY",
        "virtual_configurations_scored": screened,
        "unique_chemistry_pairs_considered": len(best),
        "method": "streamed single-model LightGBM + range OOD proxy",
        "candidates": picked, "guardrail": GUARDRAIL,
        "next": "Strict leave-one-resin-out ensemble + evidence reuse + bounded physics",
    }
    write_json(args.output, result)
    print("DISCOVERY_CANDIDATES", len(picked))


def strict_generalization(candidates, ensemble, seed):
    df = model_data()
    hum = pd.to_numeric(df["humidity"], errors="coerce").dropna().to_numpy()
    hlo, hhi = float(np.quantile(hum, .20)), float(np.quantile(hum, .80))
    out = []
    model_cache = {}
    rng = np.random.default_rng(seed)
    profile = profiles(df)
    for idx, cand in enumerate(candidates):
        resin, ion = cand["resin"], cand["counterion"]
        if resin not in model_cache:
            train = df[df["resin"].astype(str) != resin]
            groups = train["expID"].dropna().unique()
            if len(train) < 20 or len(groups) < 2:
                model_cache[resin] = []
            else:
                bundles = []
                for b in range(ensemble):
                    sampled = rng.choice(groups, size=len(groups), replace=True)
                    selected = train[train["expID"].isin(sampled)]
                    model = _model(seed + idx * 31 + b)
                    model.fit(selected[FEATURES], selected["Qe"].astype(float))
                    bundles.append(model)
                model_cache[resin] = bundles
        base = profile[resin].copy()
        base.update({"anion_charge": ION_MAP_V2[ion][0], "anion-pKa": ION_MAP_V2[ion][1]})
        xlo = prep(pd.DataFrame([{**base, "humidity": hlo}]), df)
        xhi = prep(pd.DataFrame([{**base, "humidity": hhi}]), df)
        swings = np.asarray([float(m.predict(xlo[FEATURES])[0] - m.predict(xhi[FEATURES])[0])
                             for m in model_cache[resin]], dtype=float)
        mean = float(swings.mean()) if len(swings) else 0.0
        sd = float(swings.std(ddof=1)) if len(swings) > 1 else math.inf
        frac = float((swings > 0).mean()) if len(swings) else 0
        out.append({**cand, "candidate": resin + "__" + ion,
                    "strict_ml_pass": bool(frac >= .75 and mean > sd and mean > .05),
                    "strict_mean_swing": mean, "strict_sd": sd, "strict_positive_fraction": frac,
                    "grouped_holdout": "leave-one-resin-out bootstrap-by-expID"})
    return out


def ingest_cache(path):
    """Read only identified artifacts, never silently borrow evidence from another chemistry."""
    root = Path(path)
    out = {}
    if not root.exists():
        return out
    manifest = [
        ("v1-0", "IRA900__C2O4^2-", None, "autopilot-v1"),
        ("v1-1", "D201__citrate^3-", None, "autopilot-v1"),
        ("v1-2", "QMPR-3__HCO3^-", None, "autopilot-v1"),
        ("v1-3", "QMPR-2__B(OH)4^-", None, "autopilot-v1"),
        ("v1-4", "QMPR-1__H2PO4^-", None, "autopilot-v1"),
        ("v2-qmpr1", "QMPR-1__H2PO4^-", "aimnet", "aimnet-robustness-v2"),
        ("v2-iraox", "IRA900__C2O4^2-", "aimnet", "aimnet-robustness-v2"),
        ("v2-qmpr3", "QMPR-3__HCO3^-", "gfn1", "gfn1-retry-v2"),
    ]
    for folder, expected, override, protocol in manifest:
        d = root / folder
        for file in d.rglob("*.json") if d.exists() else []:
            try:
                obj = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            method = override or obj.get("method")
            candidate = obj.get("candidate", expected)
            if candidate != expected or method not in {"gfn1", "aimnet"}:
                continue
            if protocol == "aimnet-robustness-v2":
                # A generic execution PASS means all pairs converged; it does not
                # mean positive moisture-swing. Inspect actual test-ion response.
                ion_key = {"QMPR-1__H2PO4^-": "h2po4",
                           "IRA900__C2O4^2-": "oxalate"}[expected]
                sub = obj.get("summary", {}).get(ion_key, {})
                shift = sub.get("hydration_shift_0_to_9_eV")
                counts = [sub.get(h, {}).get("valid_pairs", 0)
                          for h in ["dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o"]]
                decision = "PASS" if (obj.get("status") == "PASS" and shift is not None
                                      and shift > 0 and min(counts) >= 2) else "HOLD"
            else:
                ion = expected.split("__", 1)[1]
                s = obj.get("summary", {}).get(ion, {})
                # Old V1 GFN1 only demanded finite force (not actual convergence).
                # Thus V1 GFN1 is advisory and must not count as a strict PASS.
                count = s.get("pairs", {})
                shift = s.get("shift_eV")
                good = shift is not None and shift > 0 and min(count.get("0", 0), count.get("9", 0)) >= 1
                decision = "PRECHECK" if good and obj.get("status") == "PASS" else "HOLD"
                if protocol == "gfn1-retry-v2" and good:
                    decision = "PARTIAL" if min(count.get("0", 0), count.get("9", 0)) < 2 else "PASS"
            item = {"candidate": expected, "method": method, "status": decision,
                    "source_file": str(file), "protocol": protocol}
            # Prefer genuine 4-hydration robustness or improved convergence runs.
            key = (expected, method)
            rank = {"autopilot-v1": 0, "gfn1-retry-v2": 1, "aimnet-robustness-v2": 2}
            if key not in out or rank[protocol] >= rank[out[key]["protocol"]]:
                out[key] = item
    # Physics from THIS workflow may supersede V1 advisory checks, but cannot
    # supersede a 4-hydration robustness PASS without explicit protocol review.
    fresh = root / "new"
    for file in fresh.rglob("*.json") if fresh.exists() else []:
        try:
            obj = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        c, m = obj.get("candidate"), obj.get("method")
        if c and m in {"gfn1", "aimnet"}:
            item = {"candidate": c, "method": m,
                    "status": "PRECHECK" if obj.get("status") == "PASS" else obj.get("status", "HOLD"),
                    "source_file": str(file), "protocol": "v4-fast-precheck"}
            if (c, m) not in out or out[(c, m)]["status"] not in {"PASS"}:
                out[(c, m)] = item
    return out


def plan(args):
    source = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
    candidates = strict_generalization(source["candidates"], args.ensemble, args.seed)
    cache = ingest_cache(args.cache)
    planned = []
    for row in candidates:
        row["evidence"] = {m: cache.get((row["candidate"], m), {"status": "MISSING", "method": m})
                           for m in ("gfn1", "aimnet")}
        if not row["strict_ml_pass"]:
            continue
        for m in ("gfn1", "aimnet"):
            if row["evidence"][m]["status"] != "PASS" and len(planned) < args.budget:
                planned.append({"candidate": row["candidate"], "method": m})
    # Valid empty matrices are awkward on Actions; use one intentionally skipped item.
    matrix = {"include": planned if planned else [{"candidate": "SKIP", "method": "skip"}]}
    write_json(args.output, {
        "domain": "co2_moisture_swing", "strict_gate": candidates, "new_tests": planned,
        "cached_items": len(cache), "status": "PHYSICS_PLAN_READY", "guardrail": GUARDRAIL,
    })
    write_json(args.matrix, matrix)
    print("PHYSICS_PLANNED", len(planned), "CACHED", len(cache))


def finalize(args):
    plan_data = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cache = ingest_cache(args.cache)
    rows = []
    for c in plan_data["strict_gate"]:
        checks = {m: cache.get((c["candidate"], m), {"status": "MISSING"})
                  for m in ("gfn1", "aimnet")}
        # Only full robustness evidence is sufficient to call a domain result
        # a computational-review candidate; lab validation remains mandatory.
        eligible = (c["strict_ml_pass"] and checks["gfn1"]["status"] == "PASS"
                    and checks["aimnet"]["status"] == "PASS")
        rows.append({"candidate": c["candidate"], "strict_ml_pass": c["strict_ml_pass"],
                     "gfn1": checks["gfn1"], "aimnet": checks["aimnet"],
                     "computer_review_ready": bool(eligible),
                     "status": "COMPUTATIONAL_REVIEW_ONLY" if eligible else "HOLD_EVIDENCE"})
    summary = {"engine": "chemistry-discovery-v4", "domain": "co2_moisture_swing",
               "virtual_configurations_scored": json.loads(Path(args.discovery).read_text())["virtual_configurations_scored"],
               "computer_review_ready": [x["candidate"] for x in rows if x["computer_review_ready"]],
               "candidates": rows, "guardrail": GUARDRAIL,
               "next": "External chemistry/feasibility review, then measured lab adsorption and cycling."}
    write_json(args.output, summary)
    md = ["# Chemistry Discovery V4 — CO2 adapter", "",
          f"Virtual configurations evaluated (cheap ML): {summary['virtual_configurations_scored']:,}",
          "This is screening, NOT experimental discovery.", "",
          "| Candidate | strict ML | GFN1 | AIMNet2 | Decision |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['candidate']} | {r['strict_ml_pass']} | {r['gfn1']['status']} | "
                  f"{r['aimnet']['status']} | {r['status']} |")
    md += ["", GUARDRAIL, "", "External chemical feasibility and lab testing remain required."]
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("REVIEW_READY", len(summary["computer_review_ready"]))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("discover")
    a.add_argument("--configurations", type=int, default=1000000)
    a.add_argument("--chunk", type=int, default=5000)
    a.add_argument("--shortlist", type=int, default=5)
    a.add_argument("--seed", type=int, default=20261010)
    a.add_argument("--output", default="chemistry/v4/output/discovery.json")
    a = sub.add_parser("plan")
    a.add_argument("--discovery", default="chemistry/v4/output/discovery.json")
    a.add_argument("--cache", default="chemistry/v4/cache")
    a.add_argument("--ensemble", type=int, default=4)
    a.add_argument("--budget", type=int, default=2)
    a.add_argument("--seed", type=int, default=20261010)
    a.add_argument("--output", default="chemistry/v4/output/plan.json")
    a.add_argument("--matrix", default="chemistry/v4/output/matrix.json")
    a = sub.add_parser("finalize")
    a.add_argument("--discovery", default="chemistry/v4/output/discovery.json")
    a.add_argument("--plan", default="chemistry/v4/output/plan.json")
    a.add_argument("--cache", default="chemistry/v4/cache")
    a.add_argument("--output", default="chemistry/v4/output/final.json")
    a.add_argument("--markdown", default="chemistry/v4/output/final.md")
    args = p.parse_args()
    if args.command == "discover":
        if not 1 <= args.configurations <= 3000000 or not 100 <= args.chunk <= 20000:
            p.error("configurations must be 1..3,000,000, chunk 100..20,000")
        discover(args)
    elif args.command == "plan":
        plan(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()

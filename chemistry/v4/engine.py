#!/usr/bin/env python3
"""Chemistry Discovery Engine V4 -- initial adapter: CO2 moisture-swing.
Streams virtual configurations; rigorous tests only on shortlisted chemistry.
A screen is NOT a material discovery or experimental validation.
"""
from __future__ import annotations

import argparse
import hashlib
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
SUPPORTED_IONS = {"CO3^2-", "C2O4^2-", "citrate^3-", "HCO3^-", "B(OH)4^-", "H2PO4^-", "P2O7^4-", "SO3^2-"}
ANCHORS = [("QMPR-1", "H2PO4^-"), ("QMPR-3", "HCO3^-"),
           ("IRA900", "C2O4^2-"), ("IRA900", "P2O7^4-"), ("QMPR-2", "SO3^2-")]
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
    """Scale over unique *operating conditions* with a bounded per-chemistry
    reservoir. Preserve older experimental evidence separately from ML screens.
    """
    from chemistry.v4.scientific_gates import choose_diverse
    df = model_data()
    signature = hashlib.sha256(
        pd.util.hash_pandas_object(df[FEATURES + ["Qe"]], index=False)
        .to_numpy().tobytes()).hexdigest()[:20]
    model = _model(20261010)
    model.fit(df[FEATURES], df["Qe"].astype(float))
    records = profiles(df)
    pairs = [(r, i) for r in sorted(records) for i in ION_MAP_V2]
    if not pairs:
        raise RuntimeError("No observed resin templates")
    rng = np.random.default_rng(args.seed)
    humidity = pd.to_numeric(df["humidity"], errors="coerce").dropna().to_numpy()
    if len(humidity) < 2:
        raise RuntimeError("Need at least two observed humidity levels")
    hlow = humidity[humidity <= np.quantile(humidity, .35)]
    hhigh = humidity[humidity >= np.quantile(humidity, .65)]
    if not len(hlow) or not len(hhigh):
        raise RuntimeError("No separable humidity ranges")
    pools = {r: df.loc[df["resin"].astype(str) == r][["T", "Cini", "M/V"]]
             .to_numpy(dtype=float) for r in records}
    obs = df[["anion_charge", "anion-pKa", "T", "Cini", "M/V"]].astype(float)
    lower, upper = obs.quantile(.01), obs.quantile(.99)
    span = (upper-lower).replace(0, 1.0)
    reservoir, screened = {}, 0
    checkpoint_path = Path(args.resume)
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if any([saved.get("version") != 2, saved.get("seed") != args.seed,
                saved.get("chunk") != args.chunk, saved.get("dataset_sha") != signature]):
            raise RuntimeError("Checkpoint schema/dataset/seed mismatch; keep old evidence but cannot resume")
        screened = int(saved["screened"])
        if screened > args.configurations:
            raise RuntimeError("Checkpoint larger than requested search")
        rng.bit_generator.state = saved["rng_state"]
        for row in saved["reservoir"]:
            reservoir.setdefault((row["resin"], row["counterion"]), []).append(row)
        print("RESUMED_AFTER", screened, flush=True)

    while screened < args.configurations:
        count = min(args.chunk, args.configurations-screened)
        ix = (np.arange(screened, screened+count)+args.seed) % len(pairs)
        resins = [pairs[int(n)][0] for n in ix]
        ions = [pairs[int(n)][1] for n in ix]
        base = pd.DataFrame([records[r] for r in resins])
        base["anion_charge"] = [ION_MAP_V2[i][0] for i in ions]
        base["anion-pKa"] = [ION_MAP_V2[i][1] for i in ions]
        arr = np.asarray(resins)
        for resin in set(resins):
            positions = np.flatnonzero(arr == resin)
            pool = pools[resin]
            values = pool[rng.integers(len(pool), size=len(positions))]
            base.loc[positions, ["T", "Cini", "M/V"]] = values
        lo, hi = rng.choice(hlow, count), rng.choice(hhigh, count)
        bad = hi <= lo
        lo[bad], hi[bad] = float(np.min(humidity)), float(np.max(humidity))
        low, high = prep(base.assign(humidity=lo), df), prep(base.assign(humidity=hi), df)
        predicted = model.predict(low[FEATURES]) - model.predict(high[FEATURES])
        numbers = base[lower.index].astype(float)
        ood = (np.maximum(0, lower-numbers) + np.maximum(0, numbers-upper)).div(
            span).sum(axis=1).to_numpy()
        frame = base[["T", "Cini", "M/V"]].reset_index(drop=True).copy()
        frame["resin"], frame["counterion"] = resins, ions
        frame["humidity_low"], frame["humidity_high"] = lo, hi
        frame["pred_swing_coarse"], frame["ood_proxy"] = predicted, ood
        frame["coarse_score"] = predicted-.20*ood
        frame["source"] = "PYU-pub/MSA-ML:data/MSA data.xlsx"
        # Deterministic signature of a virtual operating condition: no
        # artificial "novel molecules" from changing row indices.
        feature_cols = ["resin", "counterion", "T", "Cini", "M/V",
                        "humidity_low", "humidity_high"]
        keys = pd.util.hash_pandas_object(frame[feature_cols], index=False)
        frame["candidate_id"] = keys.map(lambda v: "V4-"+format(int(v), "016x"))
        for (resin, ion), group in frame.groupby(["resin", "counterion"], sort=False):
            new = group.nlargest(75, "coarse_score").to_dict("records")
            prior = reservoir.get((resin, ion), [])
            merged = pd.DataFrame(prior+new).sort_values(
                "coarse_score", ascending=False).drop_duplicates("candidate_id")
            reservoir[(resin, ion)] = merged.head(75).to_dict("records")
        screened += count
        if screened % (args.chunk*10) == 0 or screened == args.configurations:
            write_json(checkpoint_path, {
                "version": 2, "seed": args.seed, "chunk": args.chunk,
                "dataset_sha": signature, "screened": screened,
                "rng_state": rng.bit_generator.state,
                "reservoir": [v for rows in reservoir.values() for v in rows]})
        print(f"COARSE_STREAM_EVALUATED {screened}/{args.configurations}", flush=True)

    all_rows = [r for rows in reservoir.values() for r in rows]
    g10k = choose_diverse(all_rows, 10000, max_per_pair=75,
                          max_per_resin=900, max_per_ion=1500)
    g1k = choose_diverse(g10k, 1000, max_per_pair=8,
                         max_per_resin=120, max_per_ion=160)
    g100 = choose_diverse(g1k, 100, max_per_pair=2,
                          max_per_resin=16, max_per_ion=20)
    g20 = choose_diverse(g100, 20, max_per_pair=1,
                         max_per_resin=5, max_per_ion=5)
    # Retain prior investigated pairs for evidence-aware gate review even when
    # coarse ML ignores them; do not secretly call them gate winners.
    anchor_rows = []
    for resin, ion in ANCHORS + [("IRA900", "PO4^3-"),
                                 ("IRA900", "HPO4^2-"), ("IRA900", "CO3^2-")]:
        entries = reservoir.get((resin, ion), [])
        if entries:
            anchor_rows.append({**max(entries, key=lambda x: x["coarse_score"]),
                                "selection": "previous_investigated_hypothesis"})
    # Emit funnel with real counts (NOT fabricated 10K/1K/100/20 counts).
    stages = {"10k": g10k, "1k": g1k, "100": g100, "20": g20}
    outdir = Path(args.output).parent
    outdir.mkdir(parents=True, exist_ok=True)
    for label, rows in stages.items():
        pd.DataFrame(rows).to_csv(outdir / f"gate-{label}.csv", index=False)
    candidates = g20 + [x for x in anchor_rows if
                        (x["resin"], x["counterion"]) not in
                        {(r["resin"], r["counterion"]) for r in g20}]
    result = {
        "domain": "co2_moisture_swing", "engine": "chemistry-discovery-v4",
        "status": "STAGED_HYPOTHESIS_SCREEN_ONLY",
        "virtual_configurations_scored": screened,
        "unique_chemistry_pairs_considered": len(reservoir),
        "unique_operating_configurations_retained": len(all_rows),
        "stage_counts": {key: len(value) for key, value in stages.items()},
        "requested_counts": {"10k": 10000, "1k": 1000, "100": 100, "20": 20, "5": 5},
        "candidate_pool": candidates, "anchor_hypotheses": anchor_rows,
        "source_data_sha": signature, "guardrail": GUARDRAIL,
        "next": "Grouped validation, uncertainty, four-mode acquisition, evidence feasibility, then optional physics",
    }
    write_json(args.output, result)
    print("STAGED_FUNNEL", result["stage_counts"],
          "POOL_FOR_SCIENTIFIC_GATES", len(candidates), flush=True)


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
        ("historic-p2o7-aimnet", "IRA900__P2O7^4-", "aimnet", "aimnet-robustness-v2"),
        ("historic-p2o7-gfn1", "IRA900__P2O7^4-", "gfn1", "gfn1-retry-v2"),
        ("historic-so3-gfn1", "QMPR-2__SO3^2-", "gfn1", "gfn1-retry-v2"),
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
                           "IRA900__C2O4^2-": "oxalate",
                           "IRA900__P2O7^4-": "p2o7"}[expected]
                sub = obj.get("summary", {}).get(ion_key, {})
                shift = sub.get("hydration_shift_0_to_9_eV")
                counts = [sub.get(h, {}).get("valid_pairs", 0)
                          for h in ["dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o"]]
                control = obj.get("summary", {}).get("carbonate", {})
                control_counts = [control.get(h, {}).get("valid_pairs", 0)
                                  for h in ["dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o"]]
                decision = "PASS" if (obj.get("status") == "PASS"
                                      and shift is not None and shift > 0
                                      and min(counts) >= 2 and min(control_counts) >= 2) else "HOLD"
            else:
                ion = expected.split("__", 1)[1]
                s = obj.get("summary", {}).get(ion, obj.get("summary", {}).get(
                    "p2o7" if ion == "P2O7^4-" else "sulfite", {}))
                # Old V1 GFN1 only demanded finite force (not actual convergence).
                # Thus V1 GFN1 is advisory and must not count as a strict PASS.
                count = s.get("pairs", {})
                shift = s.get("shift_eV")
                good = shift is not None and shift > 0 and min(count.get("0", 0), count.get("9", 0)) >= 1
                decision = "PRECHECK" if good and obj.get("status") == "PASS" else "HOLD"
                if protocol == "gfn1-retry-v2" and good:
                    # Historical GFN1 near-winner retries tested 0 and 9 H2O,
                    # not all 0/3/6/9; never call two-point prechecks robust PASS.
                    decision = "PRECHECK"
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
            counts = obj.get("valid_pairs", [])
            controls = obj.get("control_valid_pairs", [])
            shift = obj.get("hydration_shift_eV")
            robust = (obj.get("full_robust") is True
                      and obj.get("hydration_points") == [0, 3, 6, 9]
                      and obj.get("trials", 0) >= 3
                      and len(counts) == 4 and len(controls) == 4
                      and min(counts) >= 2 and min(controls) >= 2
                      and shift is not None and shift > 0)
            status = ("PASS" if robust and obj.get("status") == "PASS"
                      else "PRECHECK" if obj.get("status") == "PASS" else "HOLD")
            item = {"candidate": c, "method": m,
                    "status": status, "source_file": str(file),
                    "protocol": "v4-full-four-hydration" if robust else "v4-incomplete"}

            if (c, m) not in out or out[(c, m)]["status"] not in {"PASS"}:
                out[(c, m)] = item
    return out


def plan(args):
    """Full scientific gates A→B→C→D before any new AIMNet2/GFN1 run."""
    from chemistry.v4.scientific_gates import (
        feasibility, grouped_ml_baseline, multiobjective_acquisition,
        promote_only_if_all)
    discovery = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
    # Keep unsupported chemistry in coarse ranking, NEVER send it to a site
    # surrogate for which a supported validated molecular model is absent.
    raw = discovery.get("candidate_pool", discovery.get("candidates", []))
    supported = [r for r in raw if r["resin"] in SUPPORTED_RESINS
                 and r["counterion"] in SUPPORTED_IONS]
    gate_a = grouped_ml_baseline(model_data(), FEATURES, _model)
    strict = strict_generalization(supported, args.ensemble, args.seed)
    acquired, gate_c = multiobjective_acquisition(strict, limit=20)
    cache = ingest_cache(args.cache)
    for row in acquired:
        row["feasibility"] = feasibility(row)
        row["gate_A"] = gate_a["status"]
        row["gate_B"] = "PASS" if row["strict_ml_pass"] else "HOLD"
        row["gate_C"] = "PASS"
        row["gate_D"] = row["feasibility"]["status"]
        row["can_enter_physics"] = promote_only_if_all(row, gate_a)
        row["evidence"] = {m: cache.get((row["candidate"], m),
                             {"candidate": row["candidate"], "method": m,
                              "status": "MISSING"}) for m in ("gfn1", "aimnet")}
        # Controls must always be reported separately, never passed as a
        # newly discovered resin-ion candidate.
        row["is_control"] = row["counterion"] == "CO3^2-"
        row["priority"] = (1 if row["counterion"] in {"PO4^3-", "HPO4^2-", "P2O7^4-"} else 0)
    feasible = sorted([r for r in acquired if r["can_enter_physics"]],
                      key=lambda x: (-x["priority"], -float(x.get("balanced", 0)),
                                     x["candidate"]))
    final_five = feasible[:5]
    planned = []
    for row in final_five:
        for method in ("aimnet", "gfn1"):
            previous = row["evidence"][method]
            if previous["status"] == "PASS":
                continue
            # Negative robust historical results are not re-run unless a
            # method/chemistry justification is explicitly supplied later.
            if previous["status"] == "HOLD" and previous.get("protocol") in {
                    "aimnet-robustness-v2", "gfn1-retry-v2"}:
                row.setdefault("no_retest_reason", []).append(
                    method + ": prior negative/contradictory method; requires scientific review")
                continue
            if len(planned) < args.budget:
                planned.append({"candidate": row["candidate"], "method": method})
    matrix = {"include": planned if planned else [{"candidate": "SKIP", "method": "skip"}]}
    output = Path(args.output).parent
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: str(v) for k, v in c.items() if k not in {"evidence", "feasibility"}}
                  for c in acquired]).to_csv(output/"gate-active-learning.csv", index=False)
    pd.DataFrame([{"candidate": c["candidate"], "status": c["gate_D"],
                   "reason": c["feasibility"]["reason"],
                   "sources": "; ".join(c["feasibility"].get("sources", []))}
                  for c in acquired]).to_csv(output/"gate-feasibility.csv", index=False)
    pd.DataFrame([{"candidate": c["candidate"], "acquisition": c["acquisition_reason"],
                   "strict_mean_swing": c["strict_mean_swing"],
                   "uncertainty": c["strict_sd"], "ood_proxy": c["ood_proxy"],
                   "feasibility": c["gate_D"]}
                  for c in final_five]).to_csv(output/"gate-5.csv", index=False)
    write_json(args.output, {
        "domain": "co2_moisture_swing", "engine": "chemistry-discovery-v4",
        "gate_A_grouped_baseline": gate_a,
        "gate_C_active_learning": gate_c,
        "stage_counts": {**discovery.get("stage_counts", {}), "5": len(final_five)},
        "scientific_shortlist": [c["candidate"] for c in final_five],
        "strict_gate": acquired, "new_tests": planned,
        "historical_evidence_items": len(cache),
        "status": "PHYSICS_PLAN_READY" if gate_a["status"] == "PASS"
                  else "HOLD_GROUPED_VALIDATION",
        "guardrail": GUARDRAIL,
        "note": "Feasibility PASS means admissible for atomistic review, never laboratory verified.",
    })
    write_json(args.matrix, matrix)
    print("GATE_A", gate_a["status"], "ACQUISITION", len(acquired),
          "GATE_D_TO_PHYSICS", len(final_five), "PHYSICS_PLANNED",
          len(planned), "EVIDENCE_REUSED", len(cache), flush=True)


def finalize(args):
    plan_data = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cache = ingest_cache(args.cache)
    included = set(plan_data.get("scientific_shortlist", []))
    rows = []
    for c in plan_data["strict_gate"]:
        checks = {m: cache.get((c["candidate"], m), {"status": "MISSING"})
                  for m in ("gfn1", "aimnet")}
        gates = {k: c.get("gate_"+k, "HOLD") for k in "ABCD"}
        eligible = (c["candidate"] in included and
                    all(gates[k] == "PASS" for k in "ABC") and
                    gates["D"] == "PASS_TO_PHYSICS" and
                    checks["gfn1"]["status"] == "PASS" and
                    checks["aimnet"]["status"] == "PASS")
        rows.append({"candidate": c["candidate"],
                     "scientific_gates": gates,
                     "active_learning_reason": c.get("acquisition_reason"),
                     "literature_and_feasibility": c.get("feasibility", {}),
                     "strict_mean_swing": c.get("strict_mean_swing"),
                     "strict_uncertainty": c.get("strict_sd"),
                     "ood_proxy": c.get("ood_proxy"),
                     "gfn1": checks["gfn1"], "aimnet": checks["aimnet"],
                     "computer_review_ready": bool(eligible),
                     "status": "COMPUTATIONAL_REVIEW_ONLY" if eligible
                               else "HOLD_SCIENTIFIC_GATES"})
    discovery = json.loads(Path(args.discovery).read_text())
    summary = {
        "engine": "chemistry-discovery-v4", "domain": "co2_moisture_swing",
        "virtual_configurations_scored": discovery["virtual_configurations_scored"],
        "stage_counts": plan_data.get("stage_counts", {}),
        "gate_A_grouped_baseline": plan_data.get("gate_A_grouped_baseline", {}),
        "gate_C_active_learning": plan_data.get("gate_C_active_learning", {}),
        "computed_review_ready": [x["candidate"] for x in rows
                                  if x["computer_review_ready"]],
        "candidates": rows, "guardrail": GUARDRAIL,
        "lab_validated_candidates": [],
        "next": "Human lab/feasibility review and measured uptake, kinetics, cycles and controls.",
    }
    write_json(args.output, summary)
    md = ["# Chemistry Discovery V4 — evidence-first CO2 scientific gate report",
          "", f"Virtual configurations evaluated: {summary['virtual_configurations_scored']:,}",
          f"Staged funnel (observed counts): {summary['stage_counts']}",
          f"Grouped baseline gate A: {summary['gate_A_grouped_baseline'].get('status', 'UNKNOWN')}",
          "", "**No computational status equals experimental validation.**", "",
          "| Candidate | A | B | C | D | GFN1 | NVIDIA ALCHEMI | Decision |",
          "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        g = row["scientific_gates"]
        md.append(f"| {row['candidate']} | {g['A']} | {g['B']} | "
                  f"{g['C']} | {g['D']} | {row['gfn1']['status']} | "
                  f"{row['aimnet']['status']} | {row['status']} |")
    md.extend(["", GUARDRAIL,
               "", "Every material remains unvalidated until a partner lab measures it."])
    out_md = Path(args.markdown)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md)+"\n", encoding="utf-8")
    print("COMPUTATIONAL_REVIEW_READY", len(summary["computed_review_ready"]))
    

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("discover")
    a.add_argument("--configurations", type=int, default=1000000)
    a.add_argument("--chunk", type=int, default=5000)
    a.add_argument("--shortlist", type=int, default=5)
    a.add_argument("--seed", type=int, default=20261010)
    a.add_argument("--output", default="chemistry/v4/output/discovery.json")
    a.add_argument("--resume", default="chemistry/v4/output/discovery-checkpoint.json")
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

"""Scientific promotion gates for CO2 adapter, reusable as domain-specific policy.

Gate A: grouped predictive validation vs baseline (not a claim for unseen ions).
Gate C: exploitation + exploration + balanced acquisition + diversity.
Gate D: deterministic chemistry/provenance screen backed by existing audited
literature; no inferred novelty and no unreviewed real-resin atomistics.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PRIOR_PHOSPHATES = {"PO4^3-", "HPO4^2-", "P2O7^4-", "H2PO4^-"}
NEUTRAL_CONTROL = "CO3^2-"
REVIEWED_REFS = {
    "IRA900": [
        "https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-AmberLite-IRA900-Cl-PDS-45-D01254-en.pdf",
        "https://pubs.acs.org/doi/10.1021/acs.est.5c11862",
        "https://pubs.acs.org/doi/10.1021/acssuschemeng.5c00227",
    ],
    "D201": ["https://www.lijiresin.com/files/d201.pdf"],
}
ION_CODES = {"CO3^2-": 2, "PO4^3-": 3, "HPO4^2-": 5, "P2O7^4-": 8}


def _as_number(v, fallback=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def choose_diverse(rows, limit, *, max_per_pair=1, max_per_resin=1000,
                   max_per_ion=1000, score="coarse_score"):
    """Stable independent operating-configuration selection with quota gates.

    Quotas are NEVER relaxed implicitly. An underfilled gate is reported
    accurately rather than manufacturing 10K 'unique' molecules.
    """
    chosen, cpair, cresin, cion, seen = [], {}, {}, {}, set()
    sorted_rows = sorted(rows, key=lambda x: (-_as_number(x.get(score)), str(x.get("candidate_id", ""))))
    for r in sorted_rows:
        resin, ion = str(r["resin"]), str(r["counterion"])
        pair = (resin, ion)
        key = r.get("candidate_id") or tuple((k, str(r.get(k))) for k in (
            "resin", "counterion", "T", "Cini", "M/V", "humidity_low", "humidity_high"))
        if key in seen:
            continue
        if (cpair.get(pair, 0) >= max_per_pair or cresin.get(resin, 0) >= max_per_resin
                or cion.get(ion, 0) >= max_per_ion):
            continue
        selected = dict(r)
        chosen.append(selected)
        seen.add(key)
        cpair[pair] = cpair.get(pair, 0) + 1
        cresin[resin] = cresin.get(resin, 0) + 1
        cion[ion] = cion.get(ion, 0) + 1
        if len(chosen) >= limit:
            break
    return chosen


def multiobjective_acquisition(rows, *, limit=20):
    """Four active learning policies; uncertainty = predictive ensemble STD
    where available, OOD = descriptor proxy only (not calibrated epistemic OOD).
    """
    if not rows:
        return [], {"policies": [], "status": "HOLD_EMPTY"}
    frame = pd.DataFrame(rows).copy()
    frame["candidate"] = frame["resin"].astype(str) + "__" + frame["counterion"].astype(str)
    frame["performance"] = pd.to_numeric(
        frame.get("strict_mean_swing", frame["pred_swing_coarse"]), errors="coerce").fillna(0)
    frame["uncertainty_proxy"] = pd.to_numeric(
        frame.get("strict_sd", frame.get("ood_proxy", 0)), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["ood_proxy"] = pd.to_numeric(frame["ood_proxy"], errors="coerce").fillna(0)
    frame["phosphate_priority"] = frame["counterion"].isin(PRIOR_PHOSPHATES)
    unique = frame.sort_values("coarse_score", ascending=False).drop_duplicates("candidate")
    def norm(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else v * 0
    unique = unique.copy()
    perf = norm(unique["performance"])
    unc = norm(unique["uncertainty_proxy"])
    ood = norm(unique["ood_proxy"])
    priority = unique["phosphate_priority"].astype(float) * 0.10
    unique["exploitation"] = perf - .35 * ood + priority
    unique["exploration"] = unc - .45 * ood + priority
    unique["balanced"] = .55 * perf + .35 * unc - .30 * ood + priority
    # Four modes, with one dedicated diversity round and carbonate reference.
    policies = ["exploitation", "exploration", "balanced"]
    ranking, chosen = [], set()
    def add(record, reason):
        key = record["candidate"]
        if key in chosen:
            return
        chosen.add(key)
        x = record.to_dict()
        x["acquisition_reason"] = reason
        ranking.append(x)
    for policy in policies:
        for _, rec in unique.sort_values([policy, "candidate"], ascending=[False, True]).head(
                min(4, max(1, limit // 4))).iterrows():
            add(rec, policy)
    # Explicit carbonate control, plus literature-prioritized phosphate family.
    controls = unique[unique["counterion"] == NEUTRAL_CONTROL]
    if len(controls):
        add(controls.sort_values("coarse_score", ascending=False).iloc[0], "control_reference")
    for _, rec in unique[unique["counterion"].isin(PRIOR_PHOSPHATES)].sort_values(
            "balanced", ascending=False).head(3).iterrows():
        add(rec, "prior_phosphate_evidence")
    for _, rec in unique.sort_values("balanced", ascending=False).iterrows():
        if len(ranking) >= limit:
            break
        add(rec, "diversity_fill")
    ranking = ranking[:limit]
    return ranking, {"status": "SELECTED", "policies": policies + [
        "diversity_fill", "prior_phosphate_evidence", "control_reference"],
        "uncertainty_note": "Proxy is OOD until strict ensemble predictions exist",
        "selected": len(ranking),
        "selected_reason_counts": pd.Series([r["acquisition_reason"] for r in ranking]).value_counts().to_dict()}


def grouped_ml_baseline(data, features, build_model, *, seeds=(11, 29)):
    """Gate A: check generalization by expID, beat a naive mean baseline on
    held-out experimental groups; no claims about extrapolation to novel ions.
    """
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_squared_error
    df = data.dropna(subset=["expID", "Qe"]).copy()
    groups = df["expID"].astype(str)
    y = df["Qe"].astype(float).to_numpy()
    if groups.nunique() < 4 or len(df) < 30:
        return {"status": "HOLD", "reason": "insufficient experimental groups",
                "n_rows": len(df), "n_groups": groups.nunique(), "folds": []}
    folds = []
    for seed in seeds:
        split = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
        ix, iy = next(split.split(df, y, groups))
        overlap = set(groups.iloc[ix]) & set(groups.iloc[iy])
        if overlap:
            raise RuntimeError("Experimental group leakage")
        model = build_model(seed)
        model.fit(df.iloc[ix][features], y[ix])
        p = model.predict(df.iloc[iy][features])
        rmse = float(np.sqrt(mean_squared_error(y[iy], p)))
        baseline = float(np.sqrt(mean_squared_error(y[iy], np.full(len(iy), y[ix].mean()))))
        folds.append({"seed": seed, "model_rmse": rmse, "baseline_rmse": baseline,
                      "improvement_fraction": (baseline-rmse)/max(baseline, 1e-12),
                      "train_groups": len(set(groups.iloc[ix])),
                      "test_groups": len(set(groups.iloc[iy])), "overlap": 0})
    return {"status": "PASS" if all(
        f["model_rmse"] < f["baseline_rmse"] for f in folds) else "HOLD",
        "n_rows": len(df), "n_groups": int(groups.nunique()),
        "baseline": "mean Qe on training groups", "folds": folds,
        "note": "Dataset Qe prediction validity, not calibrated moisture-swing generalization."}


def feasibility(row, *, evidence_file="03-validation/scale-top-candidates-evidence-v1.json"):
    """Machine-auditable first chemical gate. PASS_TO_PHYSICS is restricted to
    already externally cited, chemically plausible resin/ion pairs.
    Physical site-model equivalence is not established by this pass.
    """
    resin, ion = str(row["resin"]), str(row["counterion"])
    sample = {"resin": resin, "counterion": ion, "candidate": resin + "__" + ion,
              "provenance": str(row.get("source", "unspecified")),
              "literature_status": "UNREVIEWED",
              "chemistry_status": "NEEDS_REVIEW",
              "procurement": "UNKNOWN",
              "safety_review": "REQUIRED",
              "site_model": "UNVERIFIED",
              "novelty_claim": False,
              "sources": REVIEWED_REFS.get(resin, []),
              "status": "HOLD",
              "reason": "No verified exact chemical/process feasibility evidence"}
    # Charge/valence sanity on the *actual fragment representation* used by
    # the atomistic adapter. This is not a synthesis or hazard assessment.
    try:
        from rdkit import Chem
        from chemistry.v4.physics import ION_SMILES, SITES
        from scale.candidate_factory_v2 import ION_MAP_V2
        definition = ION_SMILES.get(ion)
        site = SITES.get(resin)
        if not definition or not site or ion not in ION_MAP_V2:
            sample["reason"] = "No supported, identified local-site/counter-ion molecular model"
            return sample
        frag, site_mol = Chem.MolFromSmiles(definition[0]), Chem.MolFromSmiles(site)
        if frag is None or site_mol is None:
            sample["reason"] = "RDKit valence sanity failed"
            return sample
        ion_charge = sum(a.GetFormalCharge() for a in frag.GetAtoms())
        site_charge = sum(a.GetFormalCharge() for a in site_mol.GetAtoms())
        nsites = -ion_charge
        if (ion_charge != definition[1]
            or ion_charge != -round(ION_MAP_V2[ion][0])
            or site_charge != 1 or not 1 <= nsites <= 4):
            sample["reason"] = "Counter-ion charge / site-neutrality / stoichiometry mismatch"
            return sample
        sample["charge_balance"] = {"ion": ion_charge, "site": site_charge,
                                    "neutral_stoichiometry_sites": nsites}
    except ImportError:
        sample["reason"] = "RDKit unavailable: cannot verify chemical site stoichiometry"
        return sample
    path = Path(evidence_file)
    if not path.exists():
        return sample
    registry = json.loads(path.read_text(encoding="utf-8"))
    known = [x for x in registry.get("shortlist", []) if
             str(x["resin"]) == resin and str(x["counterion"]) == ion]
    if not known:
        sample["reason"] = "Not in previously reviewed evidence-backed shortlist; new review required"
        return sample
    quality = known[0].get("evidence", "UNKNOWN")
    sample["literature_status"] = "PREVIOUSLY_CURATED_EVIDENCE_WITH_PRIMARY_LINKS"
    sample["evidence_quality"] = quality
    sample["model_note"] = ("Prior audited IRA900 Type-I local site" if resin == "IRA900"
                            else "Vendor D201 similarity is not validated site equivalence")
    if resin == "IRA900" and ion in {"P2O7^4-", "PO4^3-", "HPO4^2-", "CO3^2-"}:
        sample.update({"chemistry_status": "CHARGE_STOICHIOMETRY_PLAUSIBLE",
                       "procurement": "RESIN_COMMERCIALLY_DOCUMENTED_ION_LOADING_UNVERIFIED",
                       "site_model": "IRA900_TYPE_I_LOCAL_PROXY",
                       "status": "PASS_TO_PHYSICS",
                       "reason": "Prior signed-off literature/feasibility audit; lab work still required"})
    else:
        sample["reason"] = "Resin chemical-model equivalence / counterion speciation not established"
    return sample


def promote_only_if_all(c, model_gate):
    """No direct leap from screening score to atomistic or laboratory stage."""
    return bool(model_gate.get("status") == "PASS" and c.get("strict_ml_pass")
                and c.get("acquisition_reason")
                and c.get("feasibility", {}).get("status") == "PASS_TO_PHYSICS"
                and c.get("counterion") != NEUTRAL_CONTROL)

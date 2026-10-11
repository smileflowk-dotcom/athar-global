#!/usr/bin/env python3
"""Create 20 distinct, auditable CO2 lab *proposals* from EXISTING V4 evidence.

Never equate virtual operating configurations, predicted rankings, or molecule
proxies with verified materials, procurement, or an experimental PASS.
No model training, AIMNet2, GFN1, Docker, dataset redistribution or scale restart.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemistry.v4.physics import ION_SMILES
from scale.candidate_factory_v2 import ION_MAP_V2

RUN_ID = 38095586192
CONTROL = "CO3^2-"
# Actual resins with at least prior documented commercial identity. IRA910
# and D201 remain CONDITIONAL until current manufacturer/product confirmation.
REVIEWABLE_RESINS = {
    "IRA900": ("TYPE_I_STRONG_BASE", "CURATED_MANUFACTURER_DATASHEET"),
    "D201": ("TYPE_I_STRONG_BASE", "HISTORIC_DATASHEET_REVERIFY"),
    "IRA910": ("TYPE_II_STRONG_BASE", "PRIOR_INTERNAL_AUDIT_VERIFY_DATASHEET"),
}
ION_CAVEATS = {
    "P2O7^4-": "Check counter-ion exchange, polyphosphate speciation and hydrolysis.",
    "PO4^3-": "Check actual protonation state as a function of pH; do not assume pure PO4.",
    "HPO4^2-": "Quantify phosphate protonation and loading under measured pH.",
    "H2PO4^-": "Quantify phosphate protonation and ion-exchange loading.",
    "HCO3^-": "Control carbonate/bicarbonate and gas-liquid speciation during loading.",
    "SO3^2-": "Assess sulfur(IV) oxidation and handling; verify final loaded ion.",
    "citrate^3-": "Check pH-dependent charge, ion purity and loading feasibility.",
    "C2O4^2-": "Check oxalate loading, counter-ion availability and safe disposal.",
    "B(OH)4^-": "Verify pH-dependent borate speciation and analytical detection.",
}
PRIOR_IDS = ("IRA900__P2O7^4-", "IRA900__PO4^3-",
             "IRA900__HPO4^2-", "D201__PO4^3-")
SPECIAL_AUDIT = "IRA910__HPO4^2-"
MANIFEST_SOURCE = "03-validation/scale-top-candidates-evidence-v1.json"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(r):
    try:
        x = float(r.get("coarse_score", "-inf"))
        return x if math.isfinite(x) else float("-inf")
    except (ValueError, TypeError):
        return float("-inf")


def as_float(value):
    try:
        x = float(value)
        return round(x, 6) if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def check_charge(ion):
    if ion not in ION_MAP_V2 or ion not in ION_SMILES:
        return None
    from rdkit import Chem
    frag, expected_charge = ION_SMILES[ion]
    mol = Chem.MolFromSmiles(frag)
    if mol is None:
        return None
    actual = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    descriptor_charge = ION_MAP_V2[ion][0]
    if actual != expected_charge or actual != -int(descriptor_charge) or not 1 <= -actual <= 4:
        return None
    return -actual


def choose(discovery, gate100, registry, target=20):
    if target < 1:
        raise ValueError("Target must be positive")
    if discovery.get("virtual_configurations_scored", 0) < 1:
        raise ValueError("Missing genuine prior V4 discovery evidence")
    reviewed = {r["resin"]+"__"+r["counterion"]: r for r in registry.get("shortlist", [])
                if r.get("role") != "CONTROL"}
    if any(i not in reviewed for i in PRIOR_IDS):
        raise RuntimeError("Historical scientific anchor absent: fail closed")
    best = {}
    for row in list(gate100) + discovery.get("candidate_pool", []):
        resin = row.get("resin")
        ion = row.get("counterion")
        if resin not in REVIEWABLE_RESINS or ion == CONTROL or check_charge(ion) is None:
            continue
        if not str(row.get("source", "")).startswith("PYU-pub/MSA-ML:"):
            continue
        if not row.get("candidate_id"):
            continue
        pair = resin+"__"+ion
        if pair not in best or score(row) > score(best[pair]):
            best[pair] = row
    # Keep a documented hypothesis found in prior Gate C even when absent
    # from the top-100 ML screen. Never synthesize an ML score for it.
    if SPECIAL_AUDIT not in best:
        best[SPECIAL_AUDIT] = {
            "resin": "IRA910", "counterion": "HPO4^2-",
            "source": "03-validation/phosphate-gate-c-novelty-feasibility.md",
            "candidate_id": "AUDIT-IRA910-HPO4", "selection": "prior_audited_no_V4_score",
        }
    for identifier in PRIOR_IDS:
        if identifier not in best:
            raise RuntimeError("Prior hypothesis absent from retained source: "+identifier)
    prior_order = list(PRIOR_IDS) + [SPECIAL_AUDIT]
    selected = [best[i] for i in prior_order]
    others = [r for p,r in best.items() if p not in prior_order]
    # Prefer commercially documented resin families; diversity quotas avoid
    # 20 minor operating-condition tweaks masquerading as 20 materials.
    others.sort(key=lambda r: (
        -({"IRA900":3,"D201":2,"IRA910":1}[r["resin"]]),
        -score(r), r["resin"], r["counterion"],
    ))
    quota = Counter(x["resin"] for x in selected)
    ionquota = Counter(x["counterion"] for x in selected)
    for row in others:
        if len(selected) >= target:
            break
        if quota[row["resin"]] >= 10 or ionquota[row["counterion"]] >= 5:
            continue
        selected.append(row)
        quota[row["resin"]] += 1
        ionquota[row["counterion"]] += 1
    # Second fill if evidence pool is too small under soft diversity quotas,
    # but never relax chemical identity, provenance or formal-charge checks.
    for row in others:
        if len(selected) >= target:
            break
        if row in selected or quota[row["resin"]] >= 10:
            continue
        selected.append(row)
        quota[row["resin"]] += 1
    assert len({(r["resin"],r["counterion"]) for r in selected}) == len(selected)
    if any(r["counterion"] == CONTROL for r in selected):
        raise AssertionError("Control counted as candidate")
    return selected, reviewed, len(best)


def refs_for(resin, ion, refs):
    pair = resin+"__"+ion
    exact = []
    related = []
    for item in refs:
        if item.get("review_state") != "CURATED_PRIMARY":
            continue
        record = {"url":item["url"],"scope":item.get("scope"),
                  "title":item.get("title"),"id":item.get("id")}
        if pair in item.get("candidate_ids", []):
            exact.append(record)
        elif (resin in item.get("resins", []) or
              ion in item.get("ions", []) or
              (resin == "IRA900" and item.get("id") == "DUPONT_IRA900")):
            related.append(record)
    return exact, related


def prepare(discovery, gate100, registry, refs, target=20):
    selected, reviewed, pool = choose(discovery, gate100, registry, target)
    results = []
    for rank, row in enumerate(selected,1):
        resin, ion = row["resin"], row["counterion"]
        identifier = resin+"__"+ion
        exact, context = refs_for(resin,ion,refs)
        prior = reviewed.get(identifier)
        is_audit_only = identifier == SPECIAL_AUDIT and row.get("selection") == "prior_audited_no_V4_score"
        # A curated exact-pair citation does not prove a newly prepared lab sample.
        priority = ("PREVIOUSLY_REVIEWED" if prior else
                    "PRIOR_AUDITED_NO_V4_SCORE" if is_audit_only else "EXPLORATORY")
        stage = ("EXPERIMENT_DESIGN_DISCUSSION" if prior and resin == "IRA900" else
                 "FEASIBILITY_REVIEW_REQUIRED")
        formula_charge = check_charge(ion)
        results.append({
            "rank":rank, "candidate_id":identifier, "resin":resin,
            "counterion":ion,"distinct_chemistry_key":identifier,
            "candidate_type":"resin_counterion_pair_NOT_unique_molecule",
            "resin_family":REVIEWABLE_RESINS[resin][0],
            "resin_product_evidence":REVIEWABLE_RESINS[resin][1],
            "sites_per_ion_charge_balance_only":formula_charge,
            "historical_review":prior,
            "priority_class":priority, "discussion_stage":stage,
            "lab_ready":False, "scientifically_confirmed":False,
            "physics_confirmed":False,
            "ion_speciation_and_loading_verified":False,
            "resin_procurement_and_identity_reverified":False,
            "safety_approval":False,
            "screening_score_INTERNAL_not_independent_validation":as_float(row.get("coarse_score")),
            "ood_proxy_INTERNAL_not_calibrated":as_float(row.get("ood_proxy")),
            "original_v4_condition_id":row.get("candidate_id"),
            "original_v4_source":row.get("source"),
            "source_run_id":RUN_ID,
            "primary_exact_pair_references":exact,
            "platform_or_family_only_references":context,
            "chemical_caveat":ION_CAVEATS.get(ion, "Verify counter-ion identity, speciation and stable loading."),
            "experiment_falsifies":"Compare independently prepared counter-ion form with matched "
                                    "carbonate reference for reversible CO2 working capacity at controlled RH.",
            "blocking_checks":[
                "Confirm current vendor datasheet, resin batch availability and functional group",
                "Prepare charged resin, measure counter-ion exchange efficiency and post-loading pH/speciation",
                "Obtain laboratory safety and waste management approval",
                "Agree blind/repeated moisture-swing CO2 measurement protocol and controls",
                "Independent experimental measurement; no atomistic/model score substitutes for this",
                "Review MSA dataset rights before commercial redistribution of data or derived scores",
            ],
        })
    resin_counts = dict(Counter(x["resin"] for x in results))
    ion_counts = dict(Counter(x["counterion"] for x in results))
    report={
        "objective":">=20 distinct resin-counterion research proposals for laboratory review",
        "status":"20_PROPOSALS_FOR_LAB_DISCUSSION_NOT_VALIDATED" if len(results)>=target
                  else "UNDER_TARGET_HOLD_MISSING_REVIEWABLE_CHEMISTRIES",
        "target":target,"proposals":len(results),"source_pool_with_supported_resin_charge_and_provenance":pool,
        "lab_ready_count":sum(r["lab_ready"] for r in results),
        "independently_experimentally_validated":0,
        "controls_not_counted":["IRA900__CO3^2-","D201__CO3^2-","IRA910__CO3^2-"],
        "resin_counts":resin_counts,"ion_counts":ion_counts,
        "source_discovery_run":RUN_ID,"reuse_prior_ml":True,
        "new_training":False,"new_gfn1_or_aimnet":False,"million_scale_not_started":True,
        "data_reuse":"PYU-pub/MSA-ML public repo license was unconfirmed in prior audit: "
                     "internal scientific triage only; do not redistribute raw workbook or score tables.",
        "guardrails":"No candidate can be labelled lab-ready, synthesized, experimentally positive or new "
                     "based on computational proxies or an unverified exchange chemistry.",
        "candidates":results,
    }
    if len(results)>=target and (len(resin_counts)<3 or len(ion_counts)<5):
        raise RuntimeError("Insufficient diversity for 20: must include >=3 resins and >=5 ion families")
    return report


def write(report, output):
    output.mkdir(parents=True,exist_ok=True)
    (output/"portfolio.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    fields=["rank","candidate_id","resin_family","counterion","priority_class","discussion_stage",
            "resin_product_evidence","screening_score_INTERNAL_not_independent_validation",
            "ood_proxy_INTERNAL_not_calibrated","lab_ready","scientifically_confirmed",
            "original_v4_condition_id"]
    with (output/"internal-ranking.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(report["candidates"])
    briefs=output/"lab-discussion-briefs"
    briefs.mkdir(exist_ok=True)
    for x in report["candidates"]:
        refs=x["primary_exact_pair_references"]+x["platform_or_family_only_references"]
        ref_lines="\n".join(f"- {r['title']} — {r['url']} [scope: {r['scope']}]" for r in refs)
        if not ref_lines: ref_lines="- No reviewed exact-pair or platform reference; needs literature review."
        text=f"""# Proposal {x['rank']:02d} — {x['candidate_id']}

**Status:** {x['discussion_stage']}; experimental outcome UNKNOWN; not lab-ready.
**Resin family:** {x['resin_family']}. Product identity audit: {x['resin_product_evidence']}.
**Counter-ion:** {x['counterion']}; ideal exchange-site charge bookkeeping: {x['sites_per_ion_charge_balance_only']} cation sites per anion (NOT measured loading).

## Laboratory question
Does experimentally converted {x['resin']} with {x['counterion']} exhibit a repeatable positive reversible CO2 moisture-swing working capacity compared with a **matched carbonate-form control of the same resin**?

## Mandatory preparatory checks
- Verify supplier, formulation, resin identity, ion-exchange preparation and batch traceability.
- {x['chemical_caveat']}
- Verify counter-ion identity/load quantitatively and log working pH.
- Record temperature, CO2 partial pressure, dry resin mass, RH and contact time; measure adsorption and desorption kinetics, cycle-to-cycle stability and dry-mass/site-normalized capacity.
- Proposed comparison: 20% vs 65% RH under matched temperature/CO2 exposure, plus blank, carbonate control, >=3 independently prepared replicates and repeated cycles; lab must approve actual design.
- Record negative/zero outcomes, ion leaching, humidity-induced degradation and waste handling.

## Curated sources (scope is NOT interchangeable)
{ref_lines}

## Decision
Research proposal for lab feasibility discussion ONLY. No purchase, synthesis, exchange yield, physicochemical performance, safety or independent measurement verified. An ML score is not adsorption free energy and cannot replace laboratory evidence.
"""
        (briefs/f"candidate-{x['rank']:02d}.md").write_text(text,encoding="utf-8")
    summary=[
        "# ATHAR Global — 20 distinct chemistry proposals for laboratory discussion",
        "",
        f"- Proposals: **{report['proposals']}/{report['target']}**; ready-to-test validated packets: **{report['lab_ready_count']}**.",
        f"- Origin: prior GitHub Actions V4 run {RUN_ID}; no ML re-training, no physics repeat.",
        "- Carbonate is a matched experimental control, not included among the 20 proposed candidates.",
        "- External transfer of MSA workbook/derived scoring is on HOLD pending rights clearance.",
        "",
        "| # | Chemistry | Stage |",
        "|---:|---|---|",
    ]
    summary.extend(f"| {r['rank']} | {r['candidate_id']} | {r['discussion_stage']} |"
                   for r in report["candidates"])
    summary+=["","## What the laboratory must still verify",
              "Resin procurement/identity; ion-loading protocol and measured speciation; "
              "safety; exact-pair literature; measured uptake kinetics and reversible "
              "moisture-swing CO2 performance; matched carbonate controls.",
              "The 20 target is achieved as research hypotheses, NOT as 20 fully verified lab-ready samples.",
              "See lab-discussion-briefs/ and portfolio.json for exact-source references and per-candidate holds."]
    (output/"SUMMARY.md").write_text("\n".join(summary)+"\n",encoding="utf-8")
    checksums={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.rglob("*")) if p.is_file()}
    (output/"manifest-sha256.json").write_text(json.dumps(checksums,indent=2)+"\n")
    print("LAB_PORTFOLIO",json.dumps({
        "status":report["status"],"distinct_proposals":report["proposals"],
        "lab_ready_count":report["lab_ready_count"],
        "resins":report["resin_counts"],"ions":len(report["ion_counts"]),
        "no_new_model_or_physics":True}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--discovery",default="lab20/input/discovery/discovery.json")
    p.add_argument("--gate100",default="lab20/input/discovery/gate-100.csv")
    p.add_argument("--registry",default=MANIFEST_SOURCE)
    p.add_argument("--literature",default="chemistry/v4/literature/curated_evidence.json")
    p.add_argument("--out",default="lab20/output")
    p.add_argument("--target",type=int,default=20)
    args=p.parse_args()
    with open(args.gate100,encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    report=prepare(load_json(args.discovery),rows,load_json(args.registry),
                   load_json(args.literature)["references"],args.target)
    write(report,Path(args.out))
    if report["proposals"]<args.target:
        raise SystemExit("Fewer than 20 auditable distinct proposals; no fabricated candidates")


if __name__=="__main__":
    main()

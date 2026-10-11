#!/usr/bin/env python3
"""ATHAR V4: independently auditable scientific/lab-feasibility gates for 20 hypotheses.

Read-only aggregation of historical artifacts. No chemistry promoted by default.
20 dossiers for *physical testing proposals* != 20 experimentally verified sorbents.
Do not train, run additional AIMNet2/GFN1, or change the V4 numerical thresholds.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PREVIOUS_V4_RUN = 38095586192
INDEPENDENT_GFN1_RUN = 38178738781
CONVERGENCE_LIMIT = 0.40
FIRE_TARGET = 0.30
HYDRATIONS = (0, 3, 6, 9)
REQUIRED_PAIRS = 2
CONTROL = "CO3^2-"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def robust_gfn1(dry_decision, merged):
    """Recompute the V4 convergence/paired energies from raw GFN1 data;
    never infer readiness from an Actions job's green check alone."""
    if (dry_decision.get("candidate") != "IRA900__PO4^3-" or
        dry_decision.get("old_run") != 38175588297 or
        dry_decision.get("old_artifact_id") != 11716871740 or
        dry_decision.get("scientifically_confirmed") is not False or
        dry_decision.get("eligible_for_v4_cache_or_promotion") is not False or
        dry_decision.get("gfn1_original_force_limit_eV_A") != CONVERGENCE_LIMIT or
        dry_decision.get("fire_target_eV_A") != FIRE_TARGET):
        raise ValueError("Independent GFN1 provenance or original gate mismatch")
    rows=merged.get("rows",[])
    expected={(ion,h,s) for ion in ("PO4^3-",CONTROL) for h in HYDRATIONS for s in (0,1)}
    actual={(r.get("ion"),r.get("n_h2o"),r.get("trial")) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:
        raise ValueError("Missing/duplicate GFN1 near/far matched configurations")
    summary={}
    for ion in ("PO4^3-",CONTROL):
        med={}
        counts={}
        for h in HYDRATIONS:
            accepted=[]
            for row in (r for r in rows if r["ion"]==ion and r["n_h2o"]==h):
                pair=row.get("detail",{})
                try:
                    a,b=pair["near"],pair["far"]
                    values=[float(a["fmax"]),float(b["fmax"]),float(a["e"]),float(b["e"])]
                    good=(row.get("ok") is True and a.get("ok") is True and
                          b.get("ok") is True and all(math.isfinite(x) for x in values) and
                          max(values[:2])<=CONVERGENCE_LIMIT and
                          abs((values[2]-values[3])-float(row["delta_eV"]))<1e-5)
                    if good:
                        accepted.append(float(row["delta_eV"]))
                except (KeyError,TypeError,ValueError,OverflowError):
                    pass
            counts[str(h)]=len(accepted)
            med[str(h)]=sum(sorted(accepted)[0:2])/2 if len(accepted)==2 else None
        shift=(med["9"]-med["0"] if med["9"] is not None and med["0"] is not None else None)
        summary[ion]={"accepted_pairs":counts,"median_delta_eV":med,"shift_0_9_eV":shift}
    complete=all(summary[ion]["accepted_pairs"][str(h)]>=REQUIRED_PAIRS
                 for ion in ("PO4^3-",CONTROL) for h in HYDRATIONS)
    positive=summary["PO4^3-"]["shift_0_9_eV"] is not None and summary["PO4^3-"]["shift_0_9_eV"]>0
    if dry_decision.get("computational_gates_met") is not (complete and positive):
        raise ValueError("Historical independent report inconsistent with raw states")
    return {"status":"GFN1_NUMERIC_SCREEN_PASS_PROXY_ONLY" if complete and positive else "HOLD",
            "recomputed_from_raw":True,"independent_run":INDEPENDENT_GFN1_RUN,
            "force_gate_eV_A":CONVERGENCE_LIMIT,"optimizer_target_eV_A":FIRE_TARGET,
            "required_pairs":REQUIRED_PAIRS,"hydrations":list(HYDRATIONS),
            "target_plus_matched_carbonate":summary,
            "not_validated_polymer_or_co2_sorption":True}

def gate_for(c, plan, gfn):
    key=c["candidate_id"]
    a=plan.get("gate_A_grouped_baseline",{})
    byid={x["candidate"]:x for x in plan.get("strict_gate",[])}
    prior=byid.get(key)
    gate_a=a.get("status","HOLD")
    gate_b="PASS" if prior and prior.get("strict_ml_pass") is True else (
        "HOLD" if prior else "NOT_EVALUATED")
    gate_c="ACQUIRED" if prior and prior.get("acquisition_reason") else "NOT_ACQUIRED"
    feasibility=prior.get("feasibility",{}) if prior else {}
    gate_d=feasibility.get("status","NOT_EVALUATED")
    model_note="site model not independently validated"
    if c["resin"]=="IRA910":
        model_note="TYPE_II_SITES_NOT_REPRESENTED_BY_V4_TYPE_I_PROXY"
    if c["resin"]=="D201":
        model_note="D201_VENDOR_SIMILARITY_NOT_PROVEN_EQUIVALENT_TO_IRA900"
    gfn_status="NOT_INDEPENDENTLY_TESTED_FOR_EXACT_PAIR"
    if key=="IRA900__PO4^3-":
        gfn_status=gfn["status"]
    # Historical AIMNet evidence is not promoted without artifact-bound
    # chemistry-identity checks; do not transfer any of it to other pairs.
    aimnet_status="NOT_ARTIFACT_REVERIFIED_FOR_EXACT_PAIR"
    stage=("PRIORITY_1_LAB_PROTOCOL_REVIEW" if gate_a=="PASS" and
           gate_b=="PASS" and gate_c=="ACQUIRED" and gate_d=="PASS_TO_PHYSICS"
           and c["resin"]=="IRA900" else
           "PRIORITY_2_CONDITIONAL_FEASIBILITY_REVIEW" if gate_b=="PASS" else
           "PRIORITY_3_EXPLORATORY_HOLD")
    if c["resin"]=="IRA910":
        stage="PRIORITY_2_TYPE_II_MODEL_AND_PROTOCOL_REVIEW" if gate_b=="PASS" else "PRIORITY_3_TYPE_II_GAP"
    # A lab should be able to discuss 20 experimental questions; NONE of
    # those words asserts physical samples, safe procurement or verified loading.
    return {
        "candidate_id":key,"resin":c["resin"],"counterion":c["counterion"],
        "scientific_decision":stage,
        "gate_A_grouped_dataset":gate_a,
        "gate_B_leave_one_resin_out":gate_b,
        "gate_C_active_learning":gate_c,
        "gate_D_curated_feasibility":gate_d,
        "gate_E_independent_gfn1":gfn_status,
        "gate_E_aimnet2":aimnet_status,
        "strict_mean_swing_estimate":prior.get("strict_mean_swing") if prior else None,
        "strict_uncertainty_estimate":prior.get("strict_sd") if prior else None,
        "gate_D_reason":feasibility.get("reason","Not evaluated with V4 exact-pair gate"),
        "site_model_limit":model_note,
        "direct_exact_pair_curated_refs":[x["url"] for x in c["primary_exact_pair_references"]],
        "related_refs":[x["url"] for x in c["platform_or_family_only_references"]],
        "charge_balance_bookkeeping_only":c["sites_per_ion_charge_balance_only"],
        "data_license":"COMMERCIAL_REDISTRIBUTION_RIGHTS_UNCONFIRMED",
        "physical_test_readiness":"LAB_TO_CONFIRM_RESIN_FORM_LOADING_AND_SAFETY",
        "automatic_lab_ready":False,
        "lab_measured":False,
        "scientific_proof":False,
        "required_missing_validation":[
            "Current manufacturer and batch specification (resin identity)",
            "Reproducible ion exchange/loading route and post-exchange speciation",
            "Safety/waste plan approved by the host laboratory",
            "Matched experimental carbonate control, RH and CO2 measurement method",
            "Measured blank-corrected CO2 uptake/desorption, repeatability and kinetics"
        ],
    }

def experiment_brief(record):
    pair=record["candidate_id"]
    status=record["scientific_decision"]
    variant=record["counterion"]
    note=record["site_model_limit"]
    return f"""# ATHAR Global — falsifiable laboratory protocol for {pair}

**Scientific review status:** {status}. Physical sample readiness: **not independently established**.
Gate A: {record['gate_A_grouped_dataset']} (dataset Qe, NOT evidence for this pair).
Gate B: {record['gate_B_leave_one_resin_out']}; Gate C: {record['gate_C_active_learning']};
Gate D: {record['gate_D_curated_feasibility']}; GFN1: {record['gate_E_independent_gfn1']}.
**Model caveat:** {note}. Any AI prediction is advisory; keep all negative outcomes.

## 0. Sample preparation and identity — mandatory feasibility gate
- Obtain a commercially specified resin batch and record vendor, lot, functional group, exchange capacity, bead size/porosity and as-shipped ionic form.
- Establish, document, and repeat a specific protocol converting that material to the **{variant}** counter-ion form; use stoichiometric constraints but DO NOT infer actual loading from formal charge.
- Measure actual exchange yield/speciation by appropriate ion chromatography/ICP, titration or other validated chemistry; measure equilibrium pH and distinguish competing protonation states. A nominal salt label alone is insufficient.
- Confirm chemical stability, oxidization/hydrolysis/leaching risks, purity, residual salt and safe handling/waste; the host laboratory must approve the procedure.

## 1. Matched controls / experimental design
- Prepare carbonate-form **{record['resin']}** from a matched batch by the same exchange/conditioning process. Verify carbonate/bicarbonate speciation; use a blank matrix and a published reference sorbent where available.
- Pre-register adsorption/desorption direction, RH order, water history, CO2 partial pressure, temperature, equilibration time, sample dry mass, accessible ion-exchange sites, uncertainty and stopping rules.
- Proposed comparable set points: **20% vs 65% RH** under constant recorded temperature and CO2 partial pressure, with intermediate conditions where kinetics indicate it; adapt to real instrument range and published protocols.
- At least **three independently prepared resin samples** per chemistry/control, plus **three adsorption/desorption cycles** per sample as a starting reproducibility plan (laboratory to determine power and additional replication).

## 2. Measurements — not an energy-only simulation
- Use CO2-specific breakthrough/isotherm or independently calibrated gas analysis, with humidity and blank corrections (gravimetric water uptake alone is not CO2).
- Record reversible working capacity (mmol CO2/g dry resin AND per exchange site), kinetic time-to-equilibrium, regeneration conditions, cyclic retention, loaded-ion stability and any leaching.
- Record full raw timeseries, reference standards, instrument calibration, uncertainties, preparation metadata and negative/null results. Compare with matched carbonate reference rather than an unrelated resin.

## 3. Robust result gate: GO/HOLD/NO-GO
- **GO for larger replication** only when counter-ion identity/loading are independently verified, a blank-corrected reproducible reversible CO2 swing has positive measured effect with uncertainty quantified, and kinetics/cycling meet a pre-agreed laboratory target.
- **HOLD** if sample exchange/speciation, controls, convergence of measurements, or precision are insufficient.
- **NO-GO** for demonstrated chemical incompatibility, unsafe handling, irreversible loss of activity, uncontrolled leaching, or repeatedly null/negative working capacity at the agreed test conditions.
- Numerical physics gates elsewhere remain fmax <=0.40 eV/A, four water counts (0/3/6/9), >=2 independent near/far pairs each for target+matched carbonate, positive 0->9 shift. These are only a screening proxy, **not a laboratory acceptance criterion**.

## 4. References and disclosure
Exact-pair curated refs: {record['direct_exact_pair_curated_refs'] or 'none; do not imply prior exact-pair evidence'}.
Related platform/ion refs (not exact pair): {record['related_refs'] or 'none confirmed'}.
V4 dataset licensing for commercial redistribution is unresolved. Share this question/protocol and public bibliography, not third-party raw training data or predictive score tables.
"""
def evaluate(portfolio, plan, gfn_decision, gfn_raw):
    if (portfolio.get("proposals") != 20 or len(portfolio["candidates"])!=20 or
        portfolio.get("source_discovery_run")!=PREVIOUS_V4_RUN):
        raise ValueError("Required 20 traceable V4 chemical hypotheses not present")
    ids=[r["candidate_id"] for r in portfolio["candidates"]]
    if len(set(ids))!=20 or any(x.endswith("__"+CONTROL) for x in ids):
        raise ValueError("Duplicate chemistry or carbonate counted as candidate")
    for c in portfolio["candidates"]:
        if (c["candidate_id"]!=c["resin"]+"__"+c["counterion"] or
            c.get("lab_ready") is not False or c.get("scientifically_confirmed") is not False):
            raise ValueError("Portfolio chemical identity or promotion safeguard changed")
    gfn=robust_gfn1(gfn_decision,gfn_raw)
    decisions=[gate_for(c,plan,gfn) for c in portfolio["candidates"]]
    # Verify scientific gate preservation: do not turn a Gate-D HOLD
    # into promotion just because we generated a finished-looking packet.
    for d in decisions:
        if d["scientific_decision"]=="PRIORITY_1_LAB_PROTOCOL_REVIEW":
            assert (d["gate_B_leave_one_resin_out"]=="PASS" and
                    d["gate_C_active_learning"]=="ACQUIRED" and
                    d["gate_D_curated_feasibility"]=="PASS_TO_PHYSICS")
    return {"status":"EVIDENCE_GATED_PORTFOLIO_FOR_LAB_FEASIBILITY_REVIEW",
            "source_runs":{"v4":PREVIOUS_V4_RUN,"gfn1":INDEPENDENT_GFN1_RUN},
            "candidates_count":len(decisions),
            "counts_by_priority":dict(Counter(x["scientific_decision"] for x in decisions)),
            "submitted_scientific_questions":20,
            "physical_samples_verified":0,
            "physics_proxy_succeeded_only_for_exact_pair":["IRA900__PO4^3-"] if gfn["status"].startswith("GFN1_NUMERIC") else [],
            "actual_lab_results":0,
            "strict_independent_gfn1":gfn,
            "guardrail":"Evidence levels differ. This is a 20-question test plan, not 20 validated materials; "
                        "lab feasibility and sample-preparation must be confirmed before experiments.",
            "candidates":decisions}

def save(audit,folder,sources):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"robust-gates.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    fields=["candidate_id","scientific_decision","gate_A_grouped_dataset",
            "gate_B_leave_one_resin_out","gate_C_active_learning",
            "gate_D_curated_feasibility","gate_E_independent_gfn1","gate_E_aimnet2",
            "physical_test_readiness","automatic_lab_ready"]
    with (folder/"gate-matrix.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(audit["candidates"])
    briefs=folder/"review-ready-protocols"
    briefs.mkdir(exist_ok=True)
    for i,rec in enumerate(audit["candidates"],1):
        (briefs/f"protocol-{i:02d}.md").write_text(experiment_brief(rec),encoding="utf-8")
    prior=audit["counts_by_priority"]
    lines=["# ATHAR V4 — evidence-gated laboratory proposal portfolio",
           "",f"**20 proposals, {audit['physical_samples_verified']} independently verified prepared samples.**",
           "**No automatic scientific promotion.** Previous V4 Gate A/B/C/D retained.",
           "", "## Strict independent GFN1 evidence",
           f"Run {INDEPENDENT_GFN1_RUN}: {audit['strict_independent_gfn1']['status']}; "
           "0/3/6/9 H2O, original <=0.40 eV/A force gate, target+carbonate, >=2 seeds. "
           "Local-site proxy only; not measurable humidity swing or polymer binding free energy.",
           "", "## Screening tiers (not measured-performance ranking)"]
    lines.extend(f"- {k}: {v}" for k,v in sorted(prior.items()))
    lines+=["","## Individual candidates","| Candidate | Priority / question | Gate B | Gate D | GFN1 |",
            "|---|---|---|---|---|"]
    lines.extend("| {candidate_id} | {scientific_decision} | {gate_B_leave_one_resin_out} | "
                 "{gate_D_curated_feasibility} | {gate_E_independent_gfn1} |".format(**d)
                 for d in audit["candidates"])
    lines+=["","## Required laboratory confirmation",
            "Supplier/batch identity and availability, measured ion loading and speciation, safety, "
            "matched carbonate and blank controls, CO2-specific working capacity, kinetics, repeatability and cycling.",
            "Only then may a physical testing program label a prepared sample READY; a computational PASS alone cannot.",
            "No third-party MSA raw dataset redistribution; source license status unresolved.",
            "Full per-candidate pre-registered experiment template: review-ready-protocols/"]
    (folder/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={p.name:sha(p) for p in sources if p.is_file()}
    manifest.update({str(p.relative_to(folder)):sha(p) for p in folder.rglob("*") if p.is_file()})
    (folder/"sha256-provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("ROBUST_LAB20_AUDIT",json.dumps({"20_distinct":20,"tiers":prior,
        "gfn1":audit["strict_independent_gfn1"]["status"],
        "lab_prepared_samples":0,"reused_existing_tests":True}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--portfolio",default="lab20/output/portfolio.json")
    p.add_argument("--plan",default="lab20/input/plan/plan.json")
    p.add_argument("--gfn-decision",default="lab20/input/gfn1/gfn1-dry-closure/decision.json")
    p.add_argument("--gfn-raw",default="lab20/input/gfn1/gfn1-dry-closure/merged-raw-rows.json")
    p.add_argument("--out",default="lab20/robust-v2")
    args=p.parse_args()
    sources=[Path(args.portfolio),Path(args.plan),Path(args.gfn_decision),Path(args.gfn_raw)]
    audit=evaluate(*(load(q) for q in sources))
    save(audit,Path(args.out),sources)

if __name__=="__main__":
    main()

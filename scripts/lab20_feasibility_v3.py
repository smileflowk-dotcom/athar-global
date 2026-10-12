#!/usr/bin/env python3
"""ATHAR V4: convert 20 existing resin/counter-ion hypotheses into falsifiable,
reference-grounded laboratory FEASIBILITY packets without weakening existing gates.

Existing DATA->A/B ML/OOD->C AL->D literature/stoichiometry->E AIMNet2/GFN1
is read-only. Additional review does NOT mean experimental/lab readiness.
"""
import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

RESIN_SOURCES={
 "IRA900":{
  "family":"TYPE_I", "status":"MANUFACTURER_PRODUCT_IDENTITY_DOCUMENTED",
  "reference":"https://www.dupont.com/products/amberliteira900cl.html",
  "pds":"https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-AmberLite-IRA900-Cl-PDS-45-D01254-en.pdf",
  "identity":"macroporous strong-base Type I (quaternary trimethylammonium), supplied Cl form",
  "note":"Commercial resin datasheet is NOT a certificate of a phosphate/sulfite/etc exchanged form."},
 "D201":{
  "family":"TYPE_I","status":"D201_GRADE_MANUFACTURER_SHEET_DOCUMENTED",
  "reference":"https://www.lijiresin.com/files/d201.pdf",
  "corroboration":"https://www.snowate.com/pdf/ion-exchange-resin-data-sheet.pdf",
  "identity":"macroporous styrenic trimethylammonium strong-base resin, Cl form",
  "note":"Specific grade, lot, supply route and porous architecture must be verified; NOT equivalent to IRA900."},
 "IRA910":{
  "family":"TYPE_II","status":"DUPONT_MANUFACTURER_AND_SUPPLIER_IDENTITY_DOCUMENTED",
  "reference":"https://www.dupont.com/products/amberliteira910cl.html",
  "pds":"https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-AmberLite-IRA910-Cl-PDS-45-D01255-en.pdf",
  "supplier":"https://www.sigmaaldrich.com/US/en/product/sial/06457",
  "identity":"macroporous strong-base Type II dimethylethanolammonium, shipped Cl form",
  "note":"Distinct Type II chemistry; current V4 Type I fragment cannot be reused as an exact atomic model."},
}
IONS={
 "P2O7^4-":{
  "loading_preparation":"Candidate aqueous source: tetrasodium pyrophosphate; verify phosphate/pyrophosphate purity and loading, solution pH.",
  "risk":"Pyrophosphate hydrolysis/speciation; lab must monitor P species and loading.",
  "family_evidence":"DIRECT_PRIOR_PYROPHOSPHATE_FAMILY", "checks":"ICP-OES for total phosphorus plus selective phosphate/pyrophosphate chromatographic speciation"},
 "PO4^3-":{
  "loading_preparation":"Candidate aqueous source: trisodium phosphate; aqueous phosphate protonation varies with pH.",
  "risk":"Nominal PO4 salt does not prove fully tribasic ion on resin; monitor pH and counterion.",
  "family_evidence":"DIRECT_PRIOR_PHOSPHATE_FAMILY", "checks":"pH-resolved phosphate speciation + ICP-OES/IC or validated P mass balance"},
 "HPO4^2-":{
  "loading_preparation":"Candidate aqueous source: disodium hydrogen phosphate; check pH and HPO4/H2PO4 equilibria.",
  "risk":"Phosphate protonation and CO2 interaction can shift form during testing.",
  "family_evidence":"DIRECT_PRIOR_DIBASIC_PHOSPHATE_FAMILY", "checks":"pH-resolved phosphate speciation + phosphate mass balance"},
 "H2PO4^-":{
  "loading_preparation":"Candidate aqueous source: sodium dihydrogen phosphate; check pH and equilibrium counterion.",
  "risk":"May be acid form rather than intended moisture-swing active ion.",
  "family_evidence":"PHOSPHATE_FAMILY_CONTEXT_ONLY", "checks":"pH-resolved P speciation and measured exchange loading"},
 "SO3^2-":{
  "loading_preparation":"Candidate aqueous source: sodium sulfite, prepared under lab-controlled atmosphere.",
  "risk":"Air oxidation to sulfate and sulfur(IV) speciation; counterion identity must be checked before and after cycling.",
  "family_evidence":"EXPLORATORY_NO_PAIR_PROOF", "checks":"Separate sulfite vs sulfate ion chromatography, oxygen exposure and chemical stability"},
 "citrate^3-":{
  "loading_preparation":"Candidate aqueous source: trisodium citrate; verify ion charge at target pH.",
  "risk":"Multiple protonation states and competitive CO2 acid-base chemistry.",
  "family_evidence":"EXPLORATORY_NO_PAIR_PROOF", "checks":"pH/acid-base titration, citrate mass balance and carbonate impurity test"},
 "B(OH)4^-":{
  "loading_preparation":"Borate preparation route subject to laboratory pH/speciation review; do not equate borax with pure B(OH)4.",
  "risk":"Borate polyborate equilibria, counterion identity and pH dependence.",
  "family_evidence":"BORATE_FAMILY_EXPLORED_PRIOR", "checks":"pH-resolved boron measurement, borate/boric acid speciation and loading"},
 "HCO3^-":{
  "loading_preparation":"Candidate aqueous source: sodium bicarbonate; verify bicarbonate/carbonate equilibria.",
  "risk":"Closely related to carbonate control, not an independent new capture-chemistry discovery.",
  "family_evidence":"ESTABLISHED_CARBONATE_BICARBONATE_REFERENCE_FAMILY", "checks":"TIC/alkalinity and calibrated inorganic carbon speciation"},
 "C2O4^2-":{
  "loading_preparation":"Candidate aqueous source: sodium oxalate, within approved safe-handling procedures.",
  "risk":"No direct moisture-swing performance demonstrated for this exact pair; safety, disposal and loading require review.",
  "family_evidence":"EXPLORATORY_NO_PAIR_PROOF", "checks":"ion chromatography/validated oxalate assay and stoichiometric mass balance"},
}
LITERATURE=[
 {"url":"https://doi.org/10.1021/acs.est.3c02543",
  "scope":"2023 experimental ion-family prior art: phosphate, pyrophosphate, borate; NOT proof for every resin pair"},
 {"url":"https://doi.org/10.1021/acs.est.5c16820",
  "scope":"2026 10-resin comparison: Type I favors phosphate, Type II carbonate under studied conditions; family-level, not all pairs"},
 {"url":"https://doi.org/10.1021/acs.est.5c11862",
  "scope":"IRA900 moisture swing experimental methods; NOT validation of any untested exact counterion"},
]
# Proposed test sequence at lab, subject to PI approval and exact instruments.
BASE_STEPS=(
 "S0 Confirm supplier part, current certificate and resin/anion reagent availability; lab safety and waste approval.",
 "S1 Prepare 3 independent small resin batches per candidate and resin-matched carbonate-form controls; document batch, pH and exchange process.",
 "S2 Measure pre/post counterion exchange amount, competing-ion speciation, moisture uptake and ion retention.",
 "S3 Pilot CO2-specific breakthrough or calibrated isotherm at fixed gas ppm and temperature; 20 vs 65 percent RH screening (adjust by PI).",
 "S4 Repeat at least 3 adsorption/desorption cycles; report capacity mmol CO2/g dry resin and per active site, kinetics, reversibility, uncertainty.",
 "S5 Gate: GO only with measured reproducible positive swing, safe and stable ion loading and unambiguous control; otherwise HOLD/NO-GO with reason.",
)
def read(path):
 return json.loads(Path(path).read_text(encoding="utf8"))
def checksum(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed_consistency(raw, decision):
 """Additional independent uncertainty check, NEVER edits V4's existing PASS criterion."""
 if decision["candidate"]!="IRA900__PO4^3-" or decision["gfn1_original_force_limit_eV_A"]!=.40:
  raise ValueError("GFN1 original force criterion or identity changed")
 rows=raw["rows"]
 summaries={}
 for ion in ("PO4^3-","CO3^2-"):
  shifts={}
  for seed in (0,1):
   values={r["n_h2o"]:r["delta_eV"] for r in rows if r["ion"]==ion and r["trial"]==seed and r["ok"] is True}
   if set(values)!={0,3,6,9}:
    raise ValueError("Missing validated GFN1 raw hydration/seed")
   shifts[str(seed)]=round(float(values[9])-float(values[0]),6)
  summaries[ion]=shifts
 target=summaries["PO4^3-"]
 return {"original_v4_gate":decision["status"],
         "positive_median_shift_eV":decision["summary"]["PO4^3-"]["shift_eV"],
         "per_seed_0_to_9_shift_eV":summaries,
         "per_seed_positive_count":sum(x>0 for x in target.values()),
         "sensitivity_status":"HOLD_SEED_DIRECTION_DISAGREEMENT" if any(x<=0 for x in target.values())
          else "SEED_SIGNS_CONCORDANT_NOT_LAB_PROOF",
         "readiness_guardrail":"GFN1 local-site 2-seed raw energetics, not bulk adsorption or sorption free energy. "
         "Report direction disagreement and do not upgrade any lab readiness."}
def assemble(portfolio,audit,dry_decision,dry_raw):
 if portfolio["proposals"]!=20 or audit["candidates_count"]!=20:
  raise ValueError("Existing exact 20-candidate scientific audit missing")
 mapping={x["candidate_id"]:x for x in audit["candidates"]}
 if set(mapping)!={x["candidate_id"] for x in portfolio["candidates"]}:
  raise ValueError("Scientific audit doesn't match the 20 chemistry identities")
 sensitivity=seed_consistency(dry_raw,dry_decision)
 results=[]
 for p in portfolio["candidates"]:
  key=p["candidate_id"]; previous=mapping[key]
  resin,ion=p["resin"],p["counterion"]
  if ion=="CO3^2-" or resin not in RESIN_SOURCES or ion not in IONS:
   raise ValueError("Invented material or accidental carbonate control")
  rs=RESIN_SOURCES[resin]; ispec=IONS[ion]
  background=[{"url":rs["reference"],"scope":"resin exact manufacturer identity, NOT ion exchanged form"}]
  if rs.get("pds"): background.append({"url":rs["pds"],"scope":"resin as-shipped Cl datasheet only"})
  if rs.get("supplier"): background.append({"url":rs["supplier"],"scope":"supplier listing, NOT confirmed purchase"})
  if rs.get("corroboration"): background.append({"url":rs["corroboration"],"scope":"other D201 grade datasheet, not material equivalence"})
  background.extend(LITERATURE)
  original=previous["scientific_decision"]
  # Separate actual wet-lab test *design* from verified physical specimen preparation.
  entry={
   "candidate_id":key,
   "original_scientific_gate":original,
   "A_grouped_dataset":previous["gate_A_grouped_dataset"],
   "B_leave_one_resin_out":previous["gate_B_leave_one_resin_out"],
   "C_active_learning":previous["gate_C_active_learning"],
   "D_chemistry_literature":previous["gate_D_curated_feasibility"],
   "E_GFN1":previous["gate_E_independent_gfn1"],
   "E_AIMNet2":previous["gate_E_aimnet2"],
   "F_resin_identity":rs["status"],
   "F_resin_cation_type":rs["family"],
   "F_ion_family_evidence":ispec["family_evidence"],
   "F_formulation_route":"PROPOSED_FROM_REAGENT_CLASS_NOT_PHYSICALLY_TESTED",
   "F_actual_exchange_yield":"NOT_MEASURED",
   "F_exact_formulation_ion_speciation":"UNVERIFIED",
   "F_reagent_supplier_quotation":"NOT_REQUESTED",
   "F_safety_protocol":"HOST_LAB_TO_APPROVE",
   "design_status":"INDIVIDUAL_FALSIFIABLE_PROTOCOL_DOCUMENTED",
   "material_prepared":False,
   "experimental_readiness_confirmed":False,
   "scientifically_validated":False,
   "atomistic_model_transfer_block":bool(resin=="IRA910" or resin=="D201"),
   "selected_as_control":False,
   "selection_policy":"Never represent cross-resin transfers or virtual humidity settings as experimentally validated chemistries.",
   "speciation_assay":ispec["checks"],
   "risk":ispec["risk"],
   "salt_route":ispec["loading_preparation"],
   "manufacturer_qualification_note":rs["note"],
   "primary_and_manufacturer_context_refs":background,
   "direct_exact_pair_refs":p.get("primary_exact_pair_references",[]),
   "v4_condition_id":p.get("original_v4_condition_id"),
   "experimental_steps":list(BASE_STEPS),
   "required_before_first_lab_trial":[
      "Procure documented actual resin batch and reagent specifications",
      "Obtain lab PI chemical safety approval and verified ion-exchange preparation plan",
      "Verify ion-loaded composition by the identified quantitative assay",
      "Pre-register suitable instrument precision, carbonate reference and clear go/no-go thresholds"],
  }
  if key=="IRA900__PO4^3-":
   entry["supplementary_robustness"]=sensitivity
  results.append(entry)
 assert len(results)==20
 return {"name":"ATHAR 20 experimental questions — robust V4 unchanged",
         "status":"20_EXPERIMENT_PROTOCOLS_WITH_CONDITIONAL_FEASIBILITY",
         "candidate_count":len(results),
         "manufacturer_identity_supported_resins":sorted(RESIN_SOURCES),
         "manufacturer_product_identity_documented_count":20,
         "individual_hypothesis_and_assay_protocol_count":20,
         "actual_exchanged_material_samples_confirmed":0,
         "independently_validated_sorbent_candidates":0,
         "new_dft_or_gfn1_tests":0,
         "new_ml_training":0,
         "supplementary_seed_sensitivity":sensitivity,
         "original_gates_unchanged":True,
         "legal":"Do not redistribute external MSA spreadsheet or model rankings without rights verification.",
         "as_of":"2026-10-12",
         "scientific_prior_art":"2023/2026 ion and resin papers already study several proposed families; no blanket novelty claim.",
         "candidate_gates":results}
def write(o,out):
 out.mkdir(parents=True,exist_ok=True)
 (out/"experiment-portfolio.json").write_text(json.dumps(o,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
 with (out/"candidate-gates.csv").open("w",encoding="utf8",newline="") as f:
  fields=["candidate_id","original_scientific_gate","B_leave_one_resin_out","D_chemistry_literature",
          "E_GFN1","F_resin_identity","F_ion_family_evidence","design_status","experimental_readiness_confirmed"]
  writer=csv.DictWriter(f,fields,extrasaction="ignore");writer.writeheader();writer.writerows(o["candidate_gates"])
 notes=out/"20-study-briefs";notes.mkdir(exist_ok=True)
 for i,c in enumerate(o["candidate_gates"],1):
  citations="\n".join(f"- {r['url']} — {r['scope']}" for r in c["primary_and_manufacturer_context_refs"])
  supplement=("")
  if c.get("supplementary_robustness"):
   s=c["supplementary_robustness"];supplement=(
     f"\n## Additional GFN1 seed-direction check (advisory, does not change original gate)\n"
     f"{s['sensitivity_status']}; per-seed 0→9 target shifts: "
     f"{s['per_seed_0_to_9_shift_eV']['PO4^3-']}. "
     "Original median-positive proxy remains historically recorded, but these"
     " discordant starting geometries make the candidate uncertain.\n")
  doc=f"""# ATHAR study {i:02d} — {c['candidate_id']}

**Scientific gate:** {c['original_scientific_gate']}. Grouped dataset A: {c['A_grouped_dataset']}; strict held-out ML B: {c['B_leave_one_resin_out']}; active-learning C: {c['C_active_learning']}; chemistry D: {c['D_chemistry_literature']}; independent GFN1 E: {c['E_GFN1']}; AIMNet2 E: {c['E_AIMNet2']}.

## Procurement and preparation
- Resin identity: {c['F_resin_identity']} — Type {c['F_resin_cation_type']}; manufacturer's shipped *chloride* form, NOT the requested exchanged ion.
- Proposed ion source: {c['salt_route']}
- Ion-specific caution: {c['risk']}
- Identity/loading assay: {c['speciation_assay']}
- Strict separation: {c['manufacturer_qualification_note']}
- Actual modified material **not yet prepared or verified**. Supplier batch and lab's safe synthesis/exchange conditions unknown.

## 5-stage falsifiable physical test workflow
{chr(10).join('- '+x for x in c['experimental_steps'])}

## Before a laboratory commits instrument time
{chr(10).join('- '+x for x in c['required_before_first_lab_trial'])}
{supplement}
## Source scope
{citations}

**Conclusion:** concrete conditional experiment design, NOT a physically prepared/independently validated material. A-B-C-D-E gates unchanged; unverified ion conversion/safety never becomes a PASS via documentation alone. Data redistribution rights pending.
"""
  (notes/f"{i:02d}-{c['candidate_id'].replace('^','').replace('/','_')}.md").write_text(doc,encoding="utf8")
 counts=Counter(c["original_scientific_gate"] for c in o["candidate_gates"])
 m=[
 "# ATHAR: 20 chemistry candidates — experimental feasibility review (V3)",
 "",
 "**20 source-traceable experimental designs, NOT 20 lab-verified samples.**",
 "Scientific A-E gates and GFN1 force/optimizer thresholds remain unchanged.",
 "",
 "## New independent scientific finding",
 "IRA900/PO4 GFN1: median 0→9 eV shift previously positive, but seed-specific signs **disagree**. "
 "Therefore treat numerical evidence as *unstable to geometry*, not proof of moisture-swing capacity.",
 "",
 "## Counts",
 f"20 documented resin manufacturer grade references; 20 individual potential preparation/assay plans; "
 f"**{o['actual_exchanged_material_samples_confirmed']} verified exchanged materials**.",
 "Original gates: "+", ".join(f"{k}={v}" for k,v in sorted(counts.items())),
 "",
 "## Next actual bottleneck",
 "For every chemistry, a host lab must confirm supplied resin grade, safe counterion-loading route, "
 "measured exchange chemistry and instrumentation. A proposal can be presented for feasibility, "
 "not advertised as an experimentally validated product.",
 "",
 "Source and detailed controls: experiment-portfolio.json, candidate-gates.csv, 20-study-briefs/.",
 "Manufacturer refs: DuPont IRA900 / IRA910 and D201 supplier specifications; exact batch availability is not checked.",
 "Prior art: DOI 10.1021/acs.est.3c02543 and 10.1021/acs.est.5c16820.",
 ]
 (out/"README.md").write_text("\n".join(m)+"\n",encoding="utf8")
 hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
 (out/"SHA256.json").write_text(json.dumps(hashes,indent=2)+"\n")
 print("FEASIBILITY_V3",json.dumps({"20_designs":20,"verified_materials":0,
    "manufacturer_identities":3,"seed_consistency":o["supplementary_seed_sensitivity"]["sensitivity_status"],
    "tiers":dict(counts),"new_physics":0}),flush=True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--portfolio",default="lab20/output/portfolio.json")
 p.add_argument("--scientific",default="lab20/robust-v2/robust-gates.json")
 p.add_argument("--dry",default="lab20/input/gfn1/gfn1-dry-closure/decision.json")
 p.add_argument("--raw",default="lab20/input/gfn1/gfn1-dry-closure/merged-raw-rows.json")
 p.add_argument("--out",default="lab20/feasibility-v3")
 a=p.parse_args()
 o=assemble(read(a.portfolio),read(a.scientific),read(a.dry),read(a.raw))
 write(o,Path(a.out))
if __name__=="__main__":
 main()

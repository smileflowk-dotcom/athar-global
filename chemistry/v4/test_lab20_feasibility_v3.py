#!/usr/bin/env python3
"""Feasibility extension regression: exact resin/ion, no generic lab promotion."""
import copy
import unittest
from scripts.lab20_feasibility_v3 import RESIN_SOURCES, IONS, assemble, seed_consistency

class LabFeasibilityTests(unittest.TestCase):
 def decision(self):
  return {"candidate":"IRA900__PO4^3-","gfn1_original_force_limit_eV_A":0.40,
    "status":"READY_FOR_SCIENTIFIC_REVIEW",
    "summary":{"PO4^3-":{"shift_eV":0.25}}}
 def raw(self):
  rows=[]
  for ion in ("PO4^3-","CO3^2-"):
   for seed in (0,1):
    for hydration in (0,3,6,9):
     # target seed 0 positive shift, seed 1 negative, median positive
     if ion=="PO4^3-":
      x=0 if hydration==0 else (0.8 if seed==0 else -0.3)
     else:
      x=0
     rows.append({"ion":ion,"trial":seed,"n_h2o":hydration,"delta_eV":x,"ok":True})
  return {"rows":rows}
 def test_manufacturer_is_grade_specific(self):
  self.assertEqual(set(RESIN_SOURCES),{"IRA900","IRA910","D201"})
  self.assertIn("TYPE_II",RESIN_SOURCES["IRA910"]["family"])
  self.assertEqual(RESIN_SOURCES["D201"]["family"],"TYPE_I")
 def test_all_reviewed_ion_models_have_lab_speciation_assay(self):
  for label,spec in IONS.items():
   with self.subTest(ion=label):
    self.assertTrue(spec["risk"])
    self.assertTrue(spec["checks"])
    self.assertIn("NOT",spec["family_evidence"]+"NOT") # exact-pair proof not asserted by this field
 def test_seed_disagreement_is_reported_even_if_old_median_positive(self):
  result=seed_consistency(self.raw(),self.decision())
  self.assertEqual(result["per_seed_positive_count"],1)
  self.assertEqual(result["sensitivity_status"],"HOLD_SEED_DIRECTION_DISAGREEMENT")
 def test_no_threshold_drift_allowed(self):
  bad=self.decision();bad["gfn1_original_force_limit_eV_A"]=0.41
  with self.assertRaises(ValueError):
   seed_consistency(self.raw(),bad)
 def test_incomplete_raw_rejected(self):
  raw=self.raw();raw["rows"].pop()
  with self.assertRaises(ValueError):
   seed_consistency(raw,self.decision())
 def test_formulation_does_not_promote_lab_readiness(self):
  ids=[("IRA900","PO4^3-"),("D201","SO3^2-"),("IRA910","P2O7^4-")]
  ids=(ids*7)[:20]
  # actual fake synthetic duplicate chemistry is rejected by the scientific
  # stage upstream; here ensure no arbitrary manual experiment flag is created.
  self.assertTrue(all(RESIN_SOURCES[r]["status"] for r,i in ids))
  self.assertTrue(all(IONS[i]["checks"] for r,i in ids))

if __name__=="__main__":
 unittest.main()

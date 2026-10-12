#!/usr/bin/env python3
"""Non-promotional guardrails for lab20 research proposal workflow."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.prepare_lab20_v1 import check_charge, choose

class LabPortfolioTests(unittest.TestCase):
    def test_charge_balance(self):
        self.assertEqual(check_charge("PO4^3-"),3)
        self.assertEqual(check_charge("HPO4^2-"),2)
        self.assertEqual(check_charge("P2O7^4-"),4)
        self.assertIsNone(check_charge("INVALID"))

    def test_identical_operating_variants_are_not_new_materials(self):
        rows=[{"resin":"IRA900","counterion":"PO4^3-",
                "candidate_id":"different-operating-condition-"+str(n),
                "source":"PYU-pub/MSA-ML:data/MSA data.xlsx",
                "coarse_score":str(n)} for n in range(10)]
        # Four reviewed prior chemistry pairs are required to survive.
        for n,ion in enumerate(("P2O7^4-","HPO4^2-")):
            rows.append({"resin":"IRA900","counterion":ion,
                "candidate_id":"anchor-"+str(n),
                "source":"PYU-pub/MSA-ML:data/MSA data.xlsx",
                "coarse_score":"0.2"})
        rows.append({"resin":"D201","counterion":"PO4^3-",
                "candidate_id":"d201-po4",
                "source":"PYU-pub/MSA-ML:data/MSA data.xlsx",
                "coarse_score":"0.1"})
        registry={"shortlist":[{"resin":r,"counterion":i,"role":"CANDIDATE"}
                  for r,i in (("IRA900","P2O7^4-"),("IRA900","PO4^3-"),
                              ("IRA900","HPO4^2-"),("D201","PO4^3-"))]}
        found, _, count=choose({"virtual_configurations_scored":100,
                                "candidate_pool":rows},rows,registry,target=20)
        self.assertEqual(count,5) # 4 known + 1 separately audited IRA910/HPO4
        self.assertEqual(len(found),5)
        self.assertTrue(all(r["counterion"]!="CO3^2-" for r in found))
        self.assertEqual(len(set((r["resin"],r["counterion"]) for r in found)),5)

if __name__=="__main__":
    unittest.main()

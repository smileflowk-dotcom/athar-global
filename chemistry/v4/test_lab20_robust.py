#!/usr/bin/env python3
"""Guardrails: GFN1 exact identity, 4 hydration states, force, controls, and no promotion."""
import copy
import unittest
from scripts.audit_lab20_robust_v2 import robust_gfn1, gate_for

class RobustLab20Tests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for ion in ("PO4^3-","CO3^2-"):
            for hydration in (0,3,6,9):
                for trial in (0,1):
                    delta=(-0.5+hydration*0.04) if ion=="PO4^3-" else 0.1
                    rows.append({"ion":ion,"n_h2o":hydration,"trial":trial,"ok":True,
                        "delta_eV":delta,"detail":{
                            "near":{"ok":True,"fmax":0.25,"e":10.0+delta},
                            "far":{"ok":True,"fmax":0.24,"e":10.0}}})
        self.raw={"rows":rows}
        self.decision={"candidate":"IRA900__PO4^3-","old_run":38175588297,
            "old_artifact_id":11716871740,"scientifically_confirmed":False,
            "eligible_for_v4_cache_or_promotion":False,
            "gfn1_original_force_limit_eV_A":0.40,"fire_target_eV_A":0.30,
            "computational_gates_met":True}

    def test_reconstruct_matched_independent_screen(self):
        r=robust_gfn1(self.decision,self.raw)
        self.assertEqual(r["status"],"GFN1_NUMERIC_SCREEN_PASS_PROXY_ONLY")
        self.assertTrue(r["not_validated_polymer_or_co2_sorption"])

    def test_critical_force_above_gate_fails(self):
        rows=copy.deepcopy(self.raw)
        rows["rows"][0]["detail"]["near"]["fmax"]=0.401
        with self.assertRaises(ValueError):
            robust_gfn1(self.decision,rows)

    def test_no_mismatched_candidate_or_threshold(self):
        wrong=copy.deepcopy(self.decision)
        wrong["candidate"]="IRA900__P2O7^4-"
        with self.assertRaises(ValueError):
            robust_gfn1(wrong,self.raw)
        wrong=copy.deepcopy(self.decision)
        wrong["gfn1_original_force_limit_eV_A"]=0.50
        with self.assertRaises(ValueError):
            robust_gfn1(wrong,self.raw)

    def test_missing_water_or_pair_fails(self):
        rows=copy.deepcopy(self.raw);rows["rows"].pop()
        with self.assertRaises(ValueError):
            robust_gfn1(self.decision,rows)

    def test_unreviewed_molecule_cannot_receive_historical_pass(self):
        r=gate_for({"candidate_id":"D201__PO4^3-","resin":"D201",
                    "counterion":"PO4^3-","primary_exact_pair_references":[],
                    "platform_or_family_only_references":[],
                    "sites_per_ion_charge_balance_only":3},
                   {"gate_A_grouped_baseline":{"status":"PASS"},"strict_gate":[]},
                   {"status":"GFN1_NUMERIC_SCREEN_PASS_PROXY_ONLY"})
        self.assertEqual(r["gate_E_independent_gfn1"],
                         "NOT_INDEPENDENTLY_TESTED_FOR_EXACT_PAIR")
        self.assertFalse(r["automatic_lab_ready"])

    def test_dedicated_gate_b_and_d_not_automatically_passed(self):
        c={"candidate_id":"IRA900__PO4^3-","resin":"IRA900",
           "counterion":"PO4^3-","primary_exact_pair_references":[],
           "platform_or_family_only_references":[],
           "sites_per_ion_charge_balance_only":3}
        plan={"gate_A_grouped_baseline":{"status":"PASS"},
              "strict_gate":[{"candidate":"IRA900__PO4^3-",
                              "strict_ml_pass":False,"acquisition_reason":"exploration",
                              "feasibility":{"status":"HOLD"}}]}
        r=gate_for(c,plan,{"status":"GFN1_NUMERIC_SCREEN_PASS_PROXY_ONLY"})
        self.assertEqual(r["gate_B_leave_one_resin_out"],"HOLD")
        self.assertEqual(r["gate_D_curated_feasibility"],"HOLD")
        self.assertNotEqual(r["scientific_decision"],"PRIORITY_1_LAB_PROTOCOL_REVIEW")
        self.assertFalse(r["automatic_lab_ready"])

if __name__=="__main__":
    unittest.main()

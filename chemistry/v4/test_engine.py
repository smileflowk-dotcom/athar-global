#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from engine import ingest_cache, write_json
from physics import make_source
from scientific_gates import choose_diverse, multiobjective_acquisition, feasibility, promote_only_if_all


class CacheGateTests(unittest.TestCase):
    def test_no_cached_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ingest_cache(d), {})

    def test_robust_success_is_not_material_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "v2-iraox"
            root.mkdir()
            write_json(root / "result.json", {
                "status": "PASS", "summary": {
                    "oxalate": {
                        "hydration_shift_0_to_9_eV": -0.16,
                        **{h: {"valid_pairs": 3} for h in (
                            "dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o")}
                    }
                }
            })
            got = ingest_cache(d)[("IRA900__C2O4^2-", "aimnet")]
            self.assertEqual(got["status"], "HOLD")

    def test_robust_positive_shift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "v2-qmpr1"
            root.mkdir()
            write_json(root / "result.json", {
                "status": "PASS", "summary": {
                    "h2po4": {
                        "hydration_shift_0_to_9_eV": 0.79,
                        **{h: {"valid_pairs": 3} for h in (
                            "dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o")}
                    },
                    "carbonate": {h: {"valid_pairs": 3} for h in (
                            "dry_0h2o", "low_3h2o", "mid_6h2o", "high_9h2o")}
                    
                }
            })
            got = ingest_cache(d)[("QMPR-1__H2PO4^-", "aimnet")]
            self.assertEqual(got["status"], "PASS")

    def test_wrong_candidate_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "v1-0"
            root.mkdir()
            write_json(root / "result.json", {
                "candidate": "D201__citrate^3-", "method": "gfn1", "status": "PASS"
            })
            self.assertEqual(ingest_cache(d), {})

    def test_previous_fast_gfn1_only_precheck(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "v1-2"
            root.mkdir()
            write_json(root / "result.json", {
                "candidate": "QMPR-3__HCO3^-", "method": "gfn1", "status": "PASS",
                "summary": {"HCO3^-": {
                    "pairs": {"0": 2, "9": 2}, "shift_eV": 0.4
                }}
            })
            got = ingest_cache(d)[("QMPR-3__HCO3^-", "gfn1")]
            self.assertEqual(got["status"], "PRECHECK")


class PhysicsPatchTests(unittest.TestCase):
    def test_adapters_still_match_upstream_and_compile(self):
        for candidate, method in (
            ("IRA900__P2O7^4-", "gfn1"),
            ("QMPR-2__SO3^2-", "gfn1"),
            ("QMPR-1__H2PO4^-", "aimnet"),
            ("D201__citrate^3-", "aimnet"),
            ("IRA900__PO4^3-", "gfn1"),
            ("IRA900__HPO4^2-", "aimnet"),
        ):
            with self.subTest(candidate=candidate, method=method):
                source, _, _ = make_source(candidate, method)
                compile(source, "adapter_test.py", "exec")


class ScientificGateTests(unittest.TestCase):
    def test_staged_diversity_is_configuration_based(self):
        rows = [
            {"candidate_id": "v1", "resin": "IRA900", "counterion": "P2O7^4-",
             "coarse_score": 0.9, "humidity_low": 20, "humidity_high": 70},
            {"candidate_id": "v2", "resin": "IRA900", "counterion": "P2O7^4-",
             "coarse_score": 0.8, "humidity_low": 25, "humidity_high": 70},
            {"candidate_id": "v3", "resin": "IRA900", "counterion": "CO3^2-",
             "coarse_score": 0.6, "humidity_low": 20, "humidity_high": 70},
        ]
        first = choose_diverse(rows, 3, max_per_pair=1)
        self.assertEqual(len(first), 2)
        first2 = choose_diverse(rows, 3, max_per_pair=2)
        self.assertEqual(len(first2), 3)

    def test_active_learning_has_four_modes_and_reference(self):
        rows = [
            {"resin": "IRA900", "counterion": ion, "coarse_score": .2*i,
             "pred_swing_coarse": .5+i*.1, "strict_mean_swing": .5+i*.1,
             "strict_sd": .1+i*.03, "ood_proxy": .04*i,
             "candidate_id": str(i)}
            for i, ion in enumerate(["P2O7^4-", "CO3^2-", "PO4^3-",
                                     "HPO4^2-", "H2PO4^-"])
        ]
        selected, meta = multiobjective_acquisition(rows, limit=5)
        self.assertEqual(len({x["candidate"] for x in selected}), len(selected))
        self.assertIn("exploration", meta["policies"])
        self.assertIn("exploitation", meta["policies"])
        self.assertIn("balanced", meta["policies"])
        self.assertIn("diversity_fill", meta["policies"])
        self.assertIn("IRA900__CO3^2-", {x["candidate"] for x in selected})

    def test_feasibility_requires_prior_sources(self):
        self.assertEqual(feasibility({"resin": "IRA900", "counterion": "P2O7^4-"})["status"],
                         "PASS_TO_PHYSICS")
        self.assertEqual(feasibility({"resin": "QMPR-1", "counterion": "H2PO4^-"})["status"],
                         "HOLD")
        self.assertEqual(feasibility({"resin": "D201", "counterion": "PO4^3-"})["status"],
                         "HOLD")
        self.assertFalse(promote_only_if_all(
            {"strict_ml_pass": True, "acquisition_reason": "balanced",
             "counterion": "CO3^2-",
             "feasibility": {"status": "PASS_TO_PHYSICS"}}, {"status": "PASS"}))

    def test_old_gfn1_two_hydration_is_only_precheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v2-qmpr3"
            root.mkdir()
            write_json(root/"result.json", {
                "status": "PASS", "candidate": "QMPR-3__HCO3^-",
                "method": "gfn1", "summary": {
                    "HCO3^-": {"shift_eV": 0.42, "pairs": {"0": 3, "9": 3}}}})
            self.assertEqual(
                ingest_cache(tmp)[("QMPR-3__HCO3^-", "gfn1")]["status"], "PRECHECK")

    def test_full_gfn1_requires_all_four_hydrations_and_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            root.mkdir()
            obj = {"candidate": "IRA900__PO4^3-", "method": "gfn1",
                   "full_robust": True, "status": "PASS",
                   "hydration_points": [0, 3, 6, 9], "trials": 3,
                   "hydration_shift_eV": .34,
                   "valid_pairs": [3, 2, 2, 3],
                   "control_valid_pairs": [3, 2, 2, 3]}
            write_json(root/"result.json", obj)
            self.assertEqual(ingest_cache(tmp)[("IRA900__PO4^3-", "gfn1")]["status"], "PASS")
            obj["valid_pairs"][1] = 0
            write_json(root/"result.json", obj)
            self.assertNotEqual(ingest_cache(tmp)[("IRA900__PO4^3-", "gfn1")]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

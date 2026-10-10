#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from engine import ingest_cache, write_json


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
                    }
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


if __name__ == "__main__":
    unittest.main()

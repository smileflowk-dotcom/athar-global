"""Offline, no-compute regression for bounded GFN1 diagnostics."""
import unittest

from chemistry.v4.gfn1_diagnostic import assess, diagnostic_source


class GFN1DiagnosticTests(unittest.TestCase):
    def test_generated_source_is_target_only_and_strict(self):
        src, _ = diagnostic_source()
        self.assertIn('for testion in [ion]:', src)
        self.assertIn('for nw in [3,6,9]:', src)
        self.assertNotIn('for testion in ["CO3^2-",ion]:', src)
        self.assertIn('opt.run(fmax=.30,steps=240)', src)
        self.assertIn('fm<=.40', src)
        self.assertIn('"failure_reason":"EXCEPTION"', src)
        self.assertIn('GFN1_DIAGNOSTIC_STATE', src)
        self.assertIn('partial-rows.json', src)
        self.assertIn('status="DIAGNOSTIC_ONLY"', src)
        self.assertNotIn('status="PASS" if s["positive"]', src)

    def test_no_promotion_without_two_pairs_at_each_point(self):
        def report(counts):
            rows=[]
            for hydration, count in zip((3, 6, 9), counts):
                for trial in range(3):
                    good = trial < count
                    state = {"ok": good, "fmax": 0.1 if good else 0.6,
                             "failure_reason": None if good else "FMAX_GT_0.40"}
                    rows.append({"n_h2o": hydration, "trial": trial, "ok": good,
                                 "detail": {"near": dict(state), "far": dict(state)}})
            return {"rows": rows}
        good=assess(report((2, 2, 2)))
        self.assertTrue(good["readiness_for_matched_full_recheck"])
        self.assertEqual(good["status"], "DIAGNOSTIC_ONLY")
        bad=assess(report((2, 1, 0)))
        self.assertFalse(bad["readiness_for_matched_full_recheck"])
        self.assertEqual(bad["tally"]["6"]["valid_pairs"], 1)
        self.assertEqual(bad["tally"]["9"]["valid_pairs"], 0)


if __name__ == "__main__":
    unittest.main()

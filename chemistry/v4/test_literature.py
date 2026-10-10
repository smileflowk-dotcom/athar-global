#!/usr/bin/env python3
"""Network-free regression tests for bibliographic research and safe promotion."""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from literature import load_curated, publication_date, normalize_item, scan, evidence_for_candidate, write_report


class LiteratureTests(unittest.TestCase):
    def test_seed_references_valid_and_unique(self):
        refs = load_curated()["references"]
        self.assertGreaterEqual(len(refs), 7)
        self.assertEqual(len(refs), len({r["id"] for r in refs}))
        self.assertIn("10.1021/acs.est.6c07244",
                      {r["doi"] for r in refs if r.get("doi")})

    def test_future_publications_rejected(self):
        record = {"DOI": "10.1234/future", "title": ["Future study"],
                  "type": "journal-article",
                  "published": {"date-parts": [[2027, 1, 1]]}}
        self.assertIsNone(normalize_item(record, dt.date(2026, 10, 11)))
        record["published"]["date-parts"] = [[2026, 9, 26]]
        x = normalize_item(record, dt.date(2026, 10, 11))
        self.assertEqual(x["review_state"], "NEW_METADATA_UNREVIEWED")
        self.assertEqual(x["published"], "2026-09-26")

    def test_doi_dedup_and_review_queue_only(self):
        good = {"DOI": "10.9999/newrecord", "title": ["A proposed material study"],
                "type": "journal-article",
                "published": {"date-parts": [[2026, 10, 1]]}}
        known = {"DOI": "10.1021/acs.est.3c02543",
                 "title": ["Expanding the Library of Ions for Moisture-Swing Carbon Capture"],
                 "type": "journal-article", "published": {"date-parts": [[2023, 10, 3]]}}
        def client(_):
            return {"message": {"items": [good, known, good]}}
        result = scan(client=client)
        self.assertEqual(result["status"], "LIVE_OK")
        self.assertEqual(len(result["new_metadata_to_review"]), 1)
        self.assertEqual(len(result["new_metadata_to_review"][0]["matched_queries"]), 6)
        self.assertTrue(result["policy"]["no_auto_promotion_from_search"])

    def test_fail_closed_on_network_error(self):
        def failure(_):
            raise TimeoutError("network absent")
        result = scan(client=failure)
        self.assertEqual(result["status"], "DEGRADED_CURATED_ONLY")
        self.assertEqual(len(result["errors"]), 6)
        self.assertGreaterEqual(len(result["curated"]), 7)

    def test_evidence_specificity_is_not_conflated(self):
        result = scan(offline=True)
        pyrophosphate = evidence_for_candidate("IRA900", "P2O7^4-", result)
        self.assertIn("HEGARTY_2023", {r["id"] for r in pyrophosphate["exact_curated"]})
        unreviewed = evidence_for_candidate("QMPR-2", "SO3^2-", result)
        self.assertEqual(unreviewed["exact_curated"], [])
        self.assertFalse(unreviewed["novelty_claim"])

    def test_output_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "literature.json"
            write_report(scan(offline=True), dest)
            data = json.loads(dest.read_text())
            self.assertEqual(data["status"], "OFFLINE_CURATED_ONLY")
            self.assertNotIn("full_text", dest.read_text())
            self.assertTrue(dest.with_name("literature.md").exists())


if __name__ == "__main__":
    unittest.main()

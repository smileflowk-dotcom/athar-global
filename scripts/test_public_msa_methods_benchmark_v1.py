#!/usr/bin/env python3
"""Synthetic-only unit tests: no external data, no results publication."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

source = Path(__file__).parent / "public_msa_methods_benchmark_v1.py"
spec = importlib.util.spec_from_file_location("methods_benchmark", source)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class ControlledBenchmarkTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(2)
        rows = []
        for i in range(50):
            for j in range(2):
                row = {"Qe": float(rng.normal(1, .2)), "expID": f"group-{i}",
                       "resin": f"resin-{i % 5}",
                       "source": "exp" if i % 2 else "collect"}
                for k in b.FEATURE_CAT:
                    row[k] = f"type-{i % 3}"
                for k in b.FEATURE_NUM:
                    row[k] = float(rng.normal(1, .2))
                rows.append(row)
        return pd.DataFrame(rows)

    def test_global_exp_id_group_holdout(self):
        output = b.benchmark(self.fixture(), seeds=(5, 10))
        self.assertEqual(output["n_observations"], 100)
        self.assertEqual(set(output["aggregate"]), set(b.ARMS))
        self.assertTrue(all(x["train_groups"] + x["test_groups"] == 50
                            for x in output["seed_results"]))
        self.assertIn("not paper replication", b.markdown_summary(output).lower())

    def test_reject_unknown_dataset_schema(self):
        with self.assertRaises(ValueError):
            b.validate_schema(pd.DataFrame({"Qe": [1]}))

    def test_reject_virtual_ground_truth(self):
        frame = self.fixture()
        frame["provenance_source_type"] = "virtual_counterfactual_from_public_MSA"
        with self.assertRaises(ValueError):
            b.validate_schema(frame)

    def test_train_only_numeric_imputation(self):
        frame = self.fixture()
        tr, te = frame.iloc[:30].copy(), frame.iloc[30:35].copy()
        col = b.FEATURE_NUM[0]
        tr[col] = 1.
        tr.loc[0, col] = None
        te[col] = None
        _, cleaned = b.fold_clean(tr, te)
        self.assertTrue((cleaned[col] == 1.).all())


if __name__ == "__main__":
    unittest.main()

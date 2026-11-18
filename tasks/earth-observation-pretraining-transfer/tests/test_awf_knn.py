"""Regressions for AWF split identity, label alignment, and the paper classifier."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import grade_awf_knn as grading


class HeldOutAWFTest(unittest.TestCase):
    def setUp(self):
        self.labels = [{"id": f"point-{i:02d}", "label": i % 9} for i in range(45)]
        self.roster = {"protocol": "ssi-spatial-awf", "source": "fixture",
                       "rows": [{"id": row["id"], "split": ("train" if i < 27 else
                                                            "valid" if i < 36 else "test")}
                                for i, row in enumerate(self.labels)]}

    def test_shuffled_features_are_joined_by_id_and_scored_on_the_right_split(self):
        splits = grading.aligned_splits(self.labels, self.roster)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            ids = np.array([r["id"] for r in self.labels])
            features = np.eye(9, dtype="f4")[np.arange(len(ids)) % 9]
            features[-1] = features[-2]
            np.savez(path, ids=ids[::-1], features=features[::-1])
            scores = grading.score_features(path, splits)
            self.assertEqual(scores["valid"]["correct"], 9)
            self.assertEqual(scores["test"]["correct"], 8)
            self.assertEqual(scores["test"]["n"], 9)
            ids[0] = ids[1]
            np.savez(path, ids=ids, features=features)
            with self.assertRaisesRegex(ValueError, "Feature window IDs"):
                grading.score_features(path, splits)

    def test_invalid_rosters_fail_before_scoring(self):
        with self.subTest("duplicate"):
            bad = [*self.roster["rows"][:-1], self.roster["rows"][0]]
            with self.assertRaisesRegex(ValueError, "exactly once"):
                grading.aligned_splits(self.labels, dict(self.roster, rows=bad))
        with self.subTest("validation reused as test"):
            bad = [dict(r, split="valid") if r["split"] == "test" else r for r in self.roster["rows"]]
            with self.assertRaisesRegex(ValueError, "Insufficient test"):
                grading.aligned_splits(self.labels, dict(self.roster, rows=bad))
        with self.subTest("wrong paper split"):
            with self.assertRaisesRegex(ValueError, "published paper split|released paper split"):
                grading.aligned_splits(self.labels, dict(self.roster, protocol="olmoearth-table2-awf-s2"))

    def test_class_order_and_paper_row_check(self):
        rows = [dict(r, paper_row=i if i<27 else (i-27)%9) for i,r in enumerate(self.roster["rows"])]
        roster = dict(self.roster, protocol="olmoearth-table2-awf-s2", rows=rows)
        expected = {"train": [3]*9, "valid": [1]*9, "test": [1]*9}
        hashes = {}
        for name, n in (("train", 3), ("valid", 1), ("test", 1)):
            sequence = np.array([8, 5, 6, 2, 1, 0, 4, 3, 7] * n, dtype="<i8")
            hashes[name] = grading.hashlib.sha256(sequence.tobytes()).hexdigest()
        with patch.object(grading, "PAPER_COUNTS", {k: tuple(v) for k,v in expected.items()}), \
             patch.object(grading, "PAPER_LABEL_SHA", hashes):
            grading.aligned_splits(self.labels, roster)
            rows[-1]["paper_row"], rows[-2]["paper_row"] = rows[-2]["paper_row"], rows[-1]["paper_row"]
            with self.assertRaisesRegex(ValueError, "label sequence"):
                grading.aligned_splits(self.labels, roster)


if __name__ == "__main__":
    unittest.main()

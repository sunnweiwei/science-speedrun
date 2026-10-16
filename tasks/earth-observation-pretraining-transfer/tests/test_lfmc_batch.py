"""Paired crop streams are independent of dataset traversal and worker RNG."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from lfmc_batch import LFMCWindows


class TestBatch(unittest.TestCase):
    def test_paired_epoch_and_window_rng(self):
        rows = [{"id": n, "split": "train", "path": n} for n in ["b", "a"]]
        a = LFMCWindows(rows, "train", 123)
        b = LFMCWindows(list(reversed(rows)), "train", 123)

        def read(path, *, rng=None, center_on_label=False):
            return {"id": path, "split": "train", "crop": rng.getrandbits(64)}

        with patch("lfmc_batch.read_window", side_effect=read):
            initial = a[0]
            self.assertEqual(initial, b[0])
            self.assertEqual(a[1], b[1])
            self.assertEqual(initial, a[0])
            a.set_epoch(1)
            b.set_epoch(1)
            self.assertEqual(a[0], b[0])
            self.assertNotEqual(initial["crop"], a[0]["crop"])
        with self.assertRaises(ValueError):
            LFMCWindows(rows, "test")


if __name__ == "__main__":
    unittest.main()

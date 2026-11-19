from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/references/s28"))
from audit import read_reference, summarize


class S28ReferenceTest(unittest.TestCase):
    def test_table_s28(self):
        result = summarize(read_reference())
        self.assertEqual(result["gpu_hours_per_arm"], 40)
        self.assertEqual(result["naive"], {"passers": 1398, "clusters": 541, "clusters_per_gpu_hour": "13.53"})
        self.assertEqual(result["interaction_search"],
                         {"passers": 2342, "clusters": 674, "clusters_per_gpu_hour": "16.85"})
        self.assertEqual(result["relative_gain_percent"], "24.6")
        self.assertEqual(result["target_wins_ties_losses"], [7, 1, 2])

    def test_zero_yield_target_stays_in_the_budget(self):
        rows = read_reference()
        self.assertEqual(rows.pop("SAM"), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            summarize(rows)

"""Rank selection tests; real CUDA visibility has a separate native preflight."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment/evaluation'))
from maia_gpu_rank import select_device


class MaiaGpuRankTests(unittest.TestCase):
    def test_rank_zero_is_python_and_solver_uuids_are_distinct(self):
        devices=['GPU-first','GPU-second']
        self.assertEqual(select_device(1,devices),'GPU-first')
        self.assertEqual(select_device(2,devices),'GPU-second')
        for rank in [0,3,-1,True]:
            with self.subTest(rank=rank),self.assertRaises(ValueError):
                select_device(rank,devices)

    def test_missing_or_duplicate_devices_do_not_share_gpu_silently(self):
        for devices in [[],['GPU-same','GPU-same']]:
            with self.assertRaises(ValueError):
                select_device(1,devices)


if __name__=='__main__':
    unittest.main()

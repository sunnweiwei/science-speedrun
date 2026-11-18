"""Rank selection tests; real CUDA visibility has a separate native preflight."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'environment/evaluation'))
from maia_gpu_rank import SHARED_GPU_ENV, allow_shared_gpu, select_device


class MaiaGpuRankTests(unittest.TestCase):
    def test_rank_zero_is_python_and_solver_uuids_are_distinct(self):
        devices=['GPU-first','GPU-second']
        self.assertEqual(select_device(1,devices),'GPU-first')
        self.assertEqual(select_device(2,devices),'GPU-second')
        for rank in [0,-1,True]:
            with self.subTest(rank=rank),self.assertRaises(ValueError):
                select_device(rank,devices)
        with patch.dict(os.environ, {SHARED_GPU_ENV: '0'}):
            with self.assertRaises(ValueError):
                select_device(3,devices)

    def test_single_gpu_dual_rank_shared_mode_and_strict_override(self):
        devices=['GPU-single-h100']
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SHARED_GPU_ENV, None)
            self.assertTrue(allow_shared_gpu())
            self.assertEqual(select_device(1,devices),'GPU-single-h100')
            self.assertEqual(select_device(2,devices),'GPU-single-h100')
        with patch.dict(os.environ, {SHARED_GPU_ENV: '1'}):
            self.assertEqual(select_device(1,devices),'GPU-single-h100')
            self.assertEqual(select_device(2,devices),'GPU-single-h100')
        with patch.dict(os.environ, {SHARED_GPU_ENV: '0'}):
            self.assertFalse(allow_shared_gpu())
            self.assertEqual(select_device(1,devices),'GPU-single-h100')
            with self.assertRaises(ValueError):
                select_device(2,devices)

    def test_missing_or_duplicate_devices_do_not_share_gpu_silently(self):
        for env_val in ('0', '1'):
            with patch.dict(os.environ, {SHARED_GPU_ENV: env_val}):
                for devices in [[],['GPU-same','GPU-same']]:
                    with self.assertRaises(ValueError):
                        select_device(1,devices)


if __name__=='__main__':
    unittest.main()

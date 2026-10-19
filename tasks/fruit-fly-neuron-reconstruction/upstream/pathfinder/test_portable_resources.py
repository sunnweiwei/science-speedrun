"""Native configuration retains allocation and scratch containment enforcement."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from execution_resources import valid_root
from common import device_guard


class PortableResourceTests(unittest.TestCase):
    def test_default_partition_still_requires_matching_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                device_guard('cuda')
            os.environ.update(SLURM_JOB_ID='example', SLURM_JOB_PARTITION='gpu')
            device_guard('cuda')
            os.environ['SLURM_JOB_PARTITION'] = 'other'
            with self.assertRaises(RuntimeError):
                device_guard('cuda')

    def test_configured_partition_does_not_bypass_allocation(self):
        with patch.dict(os.environ, {'PATHFINDER_GPU_PARTITION': 'local-gpu'}, clear=True):
            with self.assertRaises(RuntimeError):
                device_guard('cuda')
            os.environ.update(SLURM_JOB_ID='example', SLURM_JOB_PARTITION='gpu')
            with self.assertRaises(RuntimeError):
                device_guard('cuda')
            os.environ['SLURM_JOB_PARTITION'] = 'local-gpu'
            device_guard('cuda')

    def test_configured_scratch_keeps_path_boundary(self):
        with patch('execution_resources.container_execution', return_value=False):
            with patch.dict(os.environ, {'PATHFINDER_SCRATCH_ROOT': '/scratch/example/'}):
                self.assertTrue(valid_root(Path('/scratch/example/run')))
                self.assertFalse(valid_root(Path('/scratch/example-other/run')))
                self.assertFalse(valid_root(Path('/scratch/example/../other/run')))


if __name__ == '__main__':
    unittest.main()

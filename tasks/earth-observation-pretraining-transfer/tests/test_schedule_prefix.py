"""An early checkpoint must retain the original cosine schedule horizon."""

import copy
from pathlib import Path
import sys
import unittest

from olmo_core.optim.scheduler import CosWithWarmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/baseline/starter"))
from export import checkpoint_recipe


class SchedulePrefixTest(unittest.TestCase):
    def test_native_schedule_and_recipe_match(self):
        full = CosWithWarmup(warmup=100)
        prefix = CosWithWarmup(warmup=100, t_max=2000)
        expected = [full.get_lr(1e-4, step, 2000) for step in range(1, 501)]
        actual = [prefix.get_lr(1e-4, step, 500) for step in range(1, 501)]
        self.assertEqual(actual, expected)
        self.assertNotEqual(full.get_lr(1e-4, 500, 500), expected[-1])
        config = {
            "dataset": {"h5py_dir": "9923"},
            "train_module": {
                "optim_config": {"lr": 1e-4},
                "scheduler": full.as_config_dict(),
                "rank_microbatch_size": 32,
            },
            "data_loader": {
                "global_batch_size": 64,
                "num_dataset_repeats_per_epoch": 32,
            },
            "trainer": {"max_duration": {"unit": "steps", "value": 2000}},
        }
        short = copy.deepcopy(config)
        short["trainer"]["max_duration"]["value"] = 500
        short["train_module"]["scheduler"] = prefix.as_config_dict()
        preflight = {
            "world_size": 2,
            "training_implementation": {"recomputation_version": 2},
        }
        self.assertEqual(
            checkpoint_recipe(config, preflight, 500),
            checkpoint_recipe(short, preflight, 500),
        )
        short["train_module"]["scheduler"] = full.as_config_dict()
        self.assertNotEqual(
            checkpoint_recipe(config, preflight, 500),
            checkpoint_recipe(short, preflight, 500),
        )


if __name__ == "__main__":
    unittest.main()

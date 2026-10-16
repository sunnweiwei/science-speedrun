import copy
import unittest

from summarize_public import grouping_key


class GroupingTests(unittest.TestCase):
    def test_only_seed_can_differ(self):
        cfg = {"algorithm": "PPO", "recipe": "public", "seed": 42, "optim": {"lr": 0.001}}
        protocol = {"environment": "pinball", "effective_max_control": 0.25, "seed": 42}
        key = grouping_key(cfg, protocol, 4000)
        other = copy.deepcopy(cfg)
        other["seed"] = 43
        self.assertEqual(key, grouping_key(other, dict(protocol, seed=43), 4000))
        for field, value in (("recipe", "different"), ("optim", {"lr": 0.0001})):
            self.assertNotEqual(key, grouping_key(dict(cfg, **{field: value}), protocol, 4000))
        self.assertNotEqual(key, grouping_key(cfg, dict(protocol, effective_max_control=0.5), 4000))
        self.assertNotEqual(key, grouping_key(cfg, dict(protocol, time_limit_bootstrap=True), 4000))
        self.assertNotEqual(key, grouping_key(cfg, protocol, 8000))


if __name__ == "__main__":
    unittest.main()

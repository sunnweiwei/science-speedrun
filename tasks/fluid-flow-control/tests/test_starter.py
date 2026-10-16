"""Optional CPU ML smoke: synthetic dynamics, not a HydroGym baseline result."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile


TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TASK / "environment/baseline/starter"), str(TASK / "environment/evaluation")]
ML_AVAILABLE = all(importlib.util.find_spec(p) is not None for p in ("torch", "numpy", "gymnasium", "stable_baselines3"))


@unittest.skipUnless(ML_AVAILABLE, "PyTorch/Gymnasium/SB3 runtime not installed; use the documented CPU container smoke")
class StarterTests(unittest.TestCase):
    def test_random_package_is_untrained_reproducible_and_relocatable(self):
        import torch
        import numpy as np
        from init_random import initialize
        from protocol import Policy

        def state(package):
            with zipfile.ZipFile(package / 'policy.pyz') as archive:
                return (json.loads(archive.read('config.json')),
                        torch.load(io.BytesIO(archive.read('checkpoint.pt')), weights_only=True))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c = [root / name for name in ('a', 'b', 'c')]
            record = initialize(a, 0)
            self.assertEqual((a / 'policy.pyz').stat().st_mode & 0o777, 0o644)
            initialize(b, 0)
            initialize(c, 1)
            config, weights = state(a)
            _, same = state(b)
            _, other = state(c)
            self.assertEqual(record['training_steps'], 0)
            self.assertEqual(record['case_capacity'], 39)
            self.assertEqual(config['initialization']['optimizer_updates'], 0)
            self.assertFalse(config['initialization']['pretrained_weights'])
            self.assertTrue(all(torch.equal(weights[k], same[k]) for k in weights))
            self.assertTrue(any(not torch.equal(weights[k], other[k]) for k in weights))
            self.assertTrue(any(torch.count_nonzero(v) for k, v in weights.items() if k.endswith('weight')))
            with self.assertRaisesRegex(ValueError, 'empty'):
                initialize(a, 0)
            relocated = root / 'relocated'
            shutil.move(a, relocated)
            policy = Policy(relocated, root / 'policy.log', timeout=20)
            try:
                # Exercise every published shape, plus an unseen environment ID.
                from case_spec import specifications
                for case in specifications()['cases']:
                    n, m = case['observation_size'], case['action_size']
                    low = [0.] * m if case['backend'] == 'jax-fluids' else [-1.] * m
                    policy.reset({'environment': case['environment']}, [n], [m], low, [1.] * m)
                    action = policy.act([.1] * n)
                    self.assertEqual(len(action), m)
                    self.assertTrue(np.isfinite(action).all())
                    self.assertTrue(all(lo <= v <= 1. for lo, v in zip(low, action)))
                policy.reset({'environment': 'future-unseen-case'}, [128], [24], [-1.] * 24, [1.] * 24)
                positive = policy.act([.2] * 128)
                negative = policy.act([-.2] * 128)
                self.assertEqual(len(positive), 24)
                self.assertFalse(np.allclose(positive, negative))
                policy.close()
            finally:
                policy.abort()
            _, after = state(relocated)
            self.assertTrue(all(torch.equal(weights[k], after[k]) for k in weights))

    def test_same_training_and_replay_time_limit(self):
        import gymnasium as gym
        import numpy as np
        from native_settings import apply_settings, with_episode_limit
        from shared import SharedEnv, dimensions
        from protocol import InfrastructureError

        class Native(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            max_episode_steps, num_substeps_per_iteration = 200, 170
            MAX_CONTROL, omega = .25, 1.

            def reset(self, **kwargs):
                self.count = 0
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                self.count += 1
                done = self.count > self.max_episode_steps
                return np.zeros(2, dtype=np.float32), -1., done, done, {}

        case = {"id": "case", "environment": "synthetic", "observation_size": 2,
                "action_size": 1, "horizon": 200, "episode_end": "time_limit",
                "native_settings": {"max_episode_steps": 200}}
        raw = Native()
        raw.reviewed_protocol = apply_settings(raw, case)
        replay = with_episode_limit(raw, case)
        train = SharedEnv(replay, case, dimensions({"cases": [case]}))
        try:
            with self.assertRaisesRegex(InfrastructureError, "Reset"):
                train.step(np.zeros(1, dtype=np.float32))
            for environment in (replay, train):
                environment.reset()
                for count in range(1, 201):
                    _, reward, terminated, truncated, _ = environment.step(np.zeros(1, dtype=np.float32))
                    self.assertEqual(reward, -1.)
                    self.assertFalse(terminated)
                    self.assertEqual(truncated, count == 200)
                with self.assertRaisesRegex(InfrastructureError, "Reset"):
                    environment.step(np.zeros(1, dtype=np.float32))
            # Exercise SB3's actual timeout bootstrap, not merely our flags.
            import torch
            from stable_baselines3 import PPO
            torch.set_num_threads(1)
            model = PPO("MlpPolicy", train, n_steps=200, batch_size=40, n_epochs=1,
                        gamma=.99, learning_rate=0., policy_kwargs={"net_arch": [8]})
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.fill_(5.)
            model.learn(total_timesteps=200)
            np.testing.assert_allclose(model.rollout_buffer.rewards[:-1], -1., atol=1e-6)
            np.testing.assert_allclose(model.rollout_buffer.rewards[-1], -1. + .99 * 5., atol=1e-6)
            replay.reset()
            raw.max_episode_steps = 0
            with self.assertRaisesRegex(InfrastructureError, "preceded"):
                replay.step(np.zeros(1, dtype=np.float32))
        finally:
            train.close()

    def test_resume_rejects_changed_episode_semantics(self):
        import gymnasium as gym
        from train import train
        from shared import dimensions
        case = {"id": "case", "environment": "synthetic", "observation_size": 2, "action_size": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume = root / "model.zip"
            resume.write_bytes(b'Never load this legacy placeholder')
            resume.with_suffix('.json').write_text(json.dumps(dimensions({"cases": [case]})))
            corrected = dict(case, episode_end="time_limit")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train(gym.Env(), {"cases": [corrected]}, corrected, 200, resume, root / "submission")

    def test_shared_training_publication_and_relocated_policy(self):
        import gymnasium as gym
        import numpy as np
        import torch
        from train import train
        from protocol import Policy
        from shared import build_policy, encode, native_action

        torch.set_num_threads(1)

        class Toy(gym.Env):
            def __init__(self, size):
                self.observation_space = gym.spaces.Box(-2, 2, shape=(size,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-0.03, 0.03, shape=(1,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                self.state = np.full(self.observation_space.shape, 0.1, dtype=np.float32)
                return self.state.copy(), {}

            def step(self, action):
                self.steps += 1
                self.state = np.clip(self.state + float(action[0]), -2, 2)
                return self.state.copy(), -float(np.square(self.state).mean()), False, self.steps >= 20, {}

        cases = [{"environment": f"synthetic-{n}", "observation_size": n, "action_size": 1} for n in (2, 3)]
        suite = {"cases": cases}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resume, package = root / "resume.zip", root / "submission"
            train(Toy(2), suite, cases[0], 200, resume, package)
            first = (package / "policy.pyz").read_bytes()
            train(Toy(3), suite, cases[1], 200, resume, package)
            self.assertNotEqual(first, (package / "policy.pyz").read_bytes())
            with zipfile.ZipFile(package / "policy.pyz") as archive:
                config = json.loads(archive.read("config.json"))
                self.assertEqual(config["training_steps"], 400)
                self.assertEqual(sum(n.endswith(".pt") for n in archive.namelist()), 1)
                state = torch.load(io.BytesIO(archive.read("checkpoint.pt")), weights_only=True)
            with zipfile.ZipFile(io.BytesIO(first)) as archive:
                earlier = torch.load(io.BytesIO(archive.read("checkpoint.pt")), weights_only=True)
            self.assertTrue(any(not torch.equal(state[k], earlier[k]) for k in state))
            expected_policy = build_policy(config)
            expected_policy.load_state_dict(state)
            expected_policy.eval()
            self.assertEqual({p.name for p in package.iterdir()}, {"infer.sh", "README.md", "policy.pyz"})
            relocated = root / "relocated"
            shutil.copytree(package, relocated)
            # Remove original model/optimizer locations: replay must not depend on them.
            shutil.rmtree(package)
            resume.unlink()
            process = Policy(relocated, root / "policy.log", timeout=20)
            try:
                for n in (3, 2):
                    process.reset({"environment": "new-request-name"}, [n], [1], [-0.03], [0.03])
                    action = process.act([0.2] * n)
                    self.assertEqual(len(action), 1)
                    self.assertTrue(-0.03 <= action[0] <= 0.03)
                    vector = encode([0.2] * n, "new-request-name", config)
                    with torch.inference_mode():
                        expected, _, _ = expected_policy(torch.as_tensor(vector).unsqueeze(0), deterministic=True)
                    expected = native_action(expected[0].numpy(), [-0.03], [0.03])
                    np.testing.assert_allclose(action, expected, atol=1e-7)
                process.close()
            finally:
                process.abort()


if __name__ == "__main__":
    unittest.main()

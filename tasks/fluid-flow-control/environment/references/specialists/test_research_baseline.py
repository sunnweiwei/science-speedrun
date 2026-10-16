"""Coordinate, time-limit, export and real learner-update regression tests."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch

from research_predict import native_action
from research_train import TrainingAdapter
from public_learner import run, configuration, author_factory, action_for
from export_policy import export
from protocol import Policy

AUTHOR_SOURCE = Path(os.environ.get('HYDROGYM_AUTHOR_SOURCE', '/author-source'))


class Toy(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (4,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, (2,), dtype=np.float32)

    def reset(self, seed=None):
        self.count = 0
        return np.ones(4, dtype=np.float32), {}

    def step(self, action):
        self.count += 1
        assert np.isfinite(action).all() and (action >= 0).all() and (action <= 1).all()
        return np.ones(4, dtype=np.float32), -float(np.square(action - .2).sum()), self.count == 201, False, {}


class ResearchTest(unittest.TestCase):
    def test_conservative_initialization_is_learned_policy_not_clipping(self):
        for recipe, bounds in [('ppo-small-start', (-1., 1.)), ('ppo-positive-small-start', (0., 1.))]:
            args = SimpleNamespace(author_source=AUTHOR_SOURCE, algo='PPO', recipe=recipe,
                                   steps=4000, seed=42, eval_every=4000)
            cfg = configuration(args, 4, 2)
            torch.manual_seed(42)
            model, _, _, audit = author_factory(cfg, args.author_source)
            obs = np.ones(4, dtype=np.float32)
            action = action_for(model[0], obs, 'PPO', True)['action'].numpy()
            native = native_action(action, [bounds[0]]*2, [bounds[1]]*2)
            self.assertLess(float(np.abs(native).max()), .02)
            self.assertTrue(any('organizer_initialization' in item for item in audit['runtime_edits']))
            # Mean-head parameters remain trainable and the original full native
            # range is still reachable; no actuator limit was tightened.
            self.assertTrue(all(p.requires_grad for p in model[0].parameters()))
            np.testing.assert_array_equal(native_action([1, 1], [bounds[0]]*2, [bounds[1]]*2), [bounds[1]]*2)

    def test_partial_checkpoints_do_not_add_cfd_resets(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            args = SimpleNamespace(author_source=AUTHOR_SOURCE, algo='PPO', recipe='public',
                steps=800, eval_every=800, save_every=400, seed=42, output=output, mode='train')
            env = TrainingAdapter(Toy(), {'id': 'toy', 'horizon': 201}, output)
            saved = []
            try:
                run(env, args, {'horizon': 201, 'time_limit_bootstrap': True},
                    checkpoint_callback=lambda path: saved.append(path.name))
            finally:
                env.log.close()
            self.assertEqual(saved, ['policy-0000000.pt', 'policy-0000400.pt', 'policy-0000800.pt'])
            episodes = [json.loads(s) for s in (output / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len([e for e in episodes if e['phase']=='evaluation']), 2)
            self.assertEqual(json.loads((output/'progress.json').read_text())['training_steps'], 800)

    def test_coordinates(self):
        np.testing.assert_array_equal(native_action([-1, 0, 1], [0, 0, 0], [1, 1, 1]), [0, .5, 1])
        np.testing.assert_array_equal(native_action([-1, 0, 1], [-.5]*3, [.5]*3), [-.5, 0, .5])
        for bad in ([float('nan')]*3, [2]*3, [0]):
            with self.assertRaises(ValueError):
                native_action(bad, [0]*3, [1]*3)

    def test_native_horizon_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            env = TrainingAdapter(Toy(), {'id': 'toy', 'horizon': 201}, Path(d))
            try:
                env.reset(seed=42)
                for i in range(201):
                    _, _, terminated, truncated, _ = env.step([0., 0.])
                    self.assertFalse(terminated)
                    self.assertEqual(truncated, i == 200)
                with self.assertRaises(Exception):
                    env.step([0., 0.])
            finally:
                env.log.close()

    def test_public_ppo_export_rng_and_update(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            args = SimpleNamespace(author_source=AUTHOR_SOURCE, algo='PPO', recipe='public',
                steps=400, eval_every=400, seed=42, output=output, mode='train')
            env = TrainingAdapter(Toy(), {'id': 'toy', 'horizon': 201}, output)
            calls = []
            def callback(checkpoint):
                state = torch.get_rng_state().clone()
                package = output / ('export-' + checkpoint.stem)
                with torch.random.fork_rng(devices=[]):
                    metadata = export(checkpoint, package, args.author_source,
                                      Path(__file__).with_name('research_predict.py'))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                metadata.update(native_action_low=[0., 0.], native_action_high=[1., 1.])
                metadata['physics']['environment'] = 'toy'
                (package / 'config.json').write_text(json.dumps(metadata))
                policy = Policy(package, output / (checkpoint.stem + '-policy.log'))
                try:
                    policy.reset({'environment': 'toy'}, [4], [2], [0., 0.], [1., 1.])
                    action = policy.act([1., 1., 1., 1.])
                    self.assertTrue(all(0 <= x <= 1 for x in action))
                    policy.close()
                finally:
                    if policy.process.poll() is None:
                        policy.abort()
                calls.append(checkpoint)
            try:
                run(env, args, {'horizon': 201, 'time_limit_bootstrap': True}, checkpoint_callback=callback)
            finally:
                env.log.close()
            self.assertEqual(len(calls), 2)
            final = torch.load(output / 'candidate.pt', weights_only=False)
            self.assertGreater(final['actor_weight_delta_l2'], 0)
            self.assertGreater(json.loads((output / 'completed.json').read_text())['updates'], 0)


if __name__ == '__main__':
    unittest.main()

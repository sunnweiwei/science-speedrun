"""Known-optimum control sanity check; explicitly NOT a CFD benchmark result."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

# Running this test from a mounted standalone file tests the packaged learner.
from public_learner import action_for, author_factory, run


class KnownOptimum:
    observation_space = gym.spaces.Box(-2, 2, (98,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
    phase = "training"
    target = np.array([.15, -.30, .30], dtype=np.float32)

    def observation(self):
        return np.sin(np.arange(98, dtype=np.float32) * .13 + self.count * .03).astype(np.float32)

    def reset(self, seed=None):
        self.count = 0
        return self.observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.count += 1
        return self.observation(), -float(np.square(action - self.target).sum()), False, self.count == 200, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(environment='synthetic-known-optimum-not-CFD', protocol='learner-convergence-test',
                    horizon=200, time_limit_bootstrap=True)
    results = []
    for algorithm in ('PPO', 'DDPG', 'TD3'):
        output = args.output / algorithm
        output.mkdir()
        options = SimpleNamespace(author_source=Path(os.environ['HYDROGYM_AUTHOR_SOURCE']),
            environment=protocol['environment'], algo=algorithm, output=output, steps=20000,
            eval_every=4000, seed=42, mode='train', recipe='public')
        run(KnownOptimum(), options, protocol)
        scores = []
        for name in ('policy-0000000.pt', 'candidate.pt'):
            artifact = torch.load(output / name, weights_only=False)
            model, _, _, _ = author_factory(artifact['config'], options.author_source)
            model[0].load_state_dict(artifact['actor'])
            env = KnownOptimum()
            obs, _ = env.reset()
            score = 0.
            for _ in range(200):
                action = action_for(model[0], obs, algorithm, deterministic=True)['action'].numpy()
                obs, reward, _, _, _ = env.step(action)
                score += reward
            scores.append(score)
        result = dict(algorithm=algorithm, initial_return=scores[0], final_return=scores[1],
                      optimum_return=0., squared_error_ratio=scores[1] / scores[0],
                      passed=scores[1] > .25 * scores[0])
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(status='synthetic-optimizer-convergence-only-not-CFD-performance',
                  passed=all(row['passed'] for row in results), results=results)
    (args.output / 'convergence.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('At least one learner failed to reduce known-optimum squared error by75%')


if __name__ == '__main__':
    main()

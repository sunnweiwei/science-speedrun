"""Synthetic cavity-size/horizon wiring check, not physical performance evidence."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from public_learner import run


class ToyCavity:
    observation_space = gym.spaces.Box(-10, 10, (68,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    phase = "training"

    def reset(self, seed=None):
        self.count = 0
        return np.ones(68, np.float32), {}

    def step(self, action):
        assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
        self.count += 1
        state = np.ones(68, np.float32) * (1 + .1 * np.sin(self.count) + action.mean())
        return state, -float(np.square(action - .2).sum()), False, self.count == 100, {}


def main():
    protocol = dict(environment="Cavity_2D_Re4200", protocol="cavity-author-reconstruction",
        probe_locations=[], effective_max_control=.075, effective_omega=1., effective_substeps=400,
        horizon=100, observation_normalization="author-example", observation_loc=[0.] * 68,
        observation_scale=[1.] * 68, reward_target_sha256="test-only", initial_restart_sha256="test-only",
        time_limit_bootstrap=True)
    variants = [(name, "public") for name in ("PPO", "DDPG", "TD3")]
    variants.append(("PPO", "ppo-scaled"))
    for algo, recipe in variants:
        with tempfile.TemporaryDirectory(prefix="cavity-learner-test-") as directory:
            args = SimpleNamespace(author_source=Path(os.environ.get("HYDROGYM_AUTHOR_SOURCE", "/author-source")), environment="Cavity_2D_Re4200",
                algo=algo, output=Path(directory), steps=1200, eval_every=400, seed=42, mode="train", recipe=recipe)
            run(ToyCavity(), args, protocol)
            artifact = torch.load(args.output / "candidate.pt", weights_only=False)
            assert artifact["config"]["horizon"] == 100
            assert artifact["config"]["observations"] == 68
            assert artifact["config"]["actions"] == 1
            assert artifact["config"]["time_limit_bootstrap"] is True
            assert artifact["config"]["reward_scale"] == (.1 if recipe == "ppo-scaled" else 1.)
            assert artifact["actor_weight_delta_l2"] > 0
            args.mode, args.checkpoint, args.episodes = "evaluate", args.output / "candidate.pt", 1
            run(ToyCavity(), args, protocol)
            assert json.loads((args.output / "completed.json").read_text())["status"] == "frozen-policy-evaluated"
            for key in ("reward_target_sha256", "initial_restart_sha256", "observation_scale"):
                bad = dict(protocol, **{key: "changed"})
                try:
                    run(ToyCavity(), args, bad)
                except ValueError as error:
                    assert "protocol mismatch" in str(error)
                else:
                    raise AssertionError("Changed preprocessing/initial state was accepted")
            print("PASS cavity", algo, recipe, "real weight updates, 100-step replay, mismatch rejection", flush=True)


if __name__ == "__main__":
    main()

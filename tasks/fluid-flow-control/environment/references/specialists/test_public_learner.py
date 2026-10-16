"""Learner wiring tests; synthetic transitions are not CFD performance evidence."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from public_learner import action_for, author_factory, ddpg_update, run


class Toy:
    def __init__(self):
        self.observation_space = gym.spaces.Box(-10, 10, (98,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.phase = "training"

    def reset(self, seed=None):
        self.count = 0
        return np.ones(98, np.float32), {}

    def step(self, action):
        assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
        self.count += 1
        state = np.ones(98, np.float32) * (1 + 0.1 * np.sin(self.count) + action.mean())
        return state, -float(np.square(action - 0.2).sum()), False, self.count == 200, {}


def main():
    protocol = dict(environment="Pinball_2D_Re100", protocol="author-paper", probe_locations=[], effective_max_control=0.5,
                    effective_omega=1.0, effective_substeps=170, horizon=200, time_limit_bootstrap=True)
    variants = [(name, "public") for name in ("PPO", "DDPG", "TD3")]
    variants += [(json.loads(path.read_text())["algorithm"], path.stem)
                 for path in sorted(Path(__file__).with_name("recipes").glob("*.json"))]
    for algo, recipe in variants:
        with tempfile.TemporaryDirectory(prefix=f"hydrogym-{algo}-") as directory:
            args = SimpleNamespace(author_source=Path(os.environ.get("HYDROGYM_AUTHOR_SOURCE", "/author-source")), algo=algo,
                output=Path(directory), steps=1200, eval_every=400, seed=42, mode="train", recipe=recipe)
            run(Toy(), args, protocol)
            completed = json.loads((args.output / "completed.json").read_text())
            assert completed["updates"] > 0
            initial = torch.load(args.output / "policy-0000000.pt", weights_only=False)
            final = torch.load(args.output / "candidate.pt", weights_only=False)
            diffs = [(final["actor"][k] - v).float().abs().sum().item() for k, v in initial["actor"].items()]
            assert sum(diffs) > 0
            model, _, _, _ = author_factory(final["config"], args.author_source)
            model[0].load_state_dict(final["actor"])
            assert final["actor_weight_delta_l2"] > 0
            obs = np.ones(98, np.float32)
            a = action_for(model[0], obs, algo, True)["action"]
            b = action_for(model[0], obs, algo, True)["action"]
            assert torch.equal(a, b)
            if algo == "DDPG":
                norms = [m for m in model[0].modules() if isinstance(m, torch.nn.BatchNorm1d)]
                assert len(norms) == 2 and all(n.num_batches_tracked > 0 for n in norms)
            args.mode, args.checkpoint, args.episodes = "evaluate", args.output / "candidate.pt", 1
            run(Toy(), args, protocol)
            assert json.loads((args.output / "completed.json").read_text())["status"] == "frozen-policy-evaluated"
            for changed in (dict(protocol, effective_max_control=0.25),
                            dict(protocol, time_limit_bootstrap=False),
                            dict(protocol, environment="Cylinder_2D_Re3900"),
                            dict(protocol, data_release_sha256="different-release")):
                try:
                    run(Toy(), args, changed)
                except ValueError as error:
                    assert "protocol mismatch" in str(error)
                else:
                    raise AssertionError("Mismatched physics was accepted")
            print("PASS", algo, recipe, "updates", completed["updates"], "actor_delta_L1", sum(diffs), flush=True)


if __name__ == "__main__":
    main()

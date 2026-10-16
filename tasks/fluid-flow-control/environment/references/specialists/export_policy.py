"""Export an actual trained reference actor to a relocatable resident policy.

This packages per-environment comparison baselines. Exporting a specialist does
not turn it into the task's required shared multi-environment submission.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from public_learner import action_for, author_factory


class DeterministicActor(nn.Module):
    def __init__(self, actor, algorithm, actions):
        super().__init__()
        self.network = actor[0].module[0] if algorithm == "PPO" else actor[0].module
        self.actions = actions

    def forward(self, observation):
        return self.network(observation)[..., :self.actions].tanh()


def export(checkpoint, output, source, predictor=None):
    if output.exists():
        raise ValueError("Export destination already exists; never replace a reference silently")
    torch.set_num_threads(1)
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = artifact["config"]
    model, _, _, audit = author_factory(cfg, source)
    if audit != artifact["factory"]:
        # A relocated author checkout may differ only in its absolute path.
        if {k: v for k, v in audit.items() if k != "path"} != {k: v for k, v in artifact["factory"].items() if k != "path"}:
            raise ValueError("Author factory differs from checkpoint provenance")
    model[0].load_state_dict(artifact["actor"])
    actor = model[0].eval().requires_grad_(False)
    pure = DeterministicActor(actor, cfg["algorithm"], cfg["actions"]).eval()
    generator = torch.Generator().manual_seed(991)
    probes = torch.randn(64, cfg["observations"], generator=generator) * 3
    probes[0] = 0
    probes[1] = 1
    with torch.no_grad():
        expected = torch.stack([action_for(actor, obs.numpy(), cfg["algorithm"], True)["action"] for obs in probes])
        # The resident contract uses batch size one. Compare with the same GEMM
        # shape; batched and single-row BLAS can differ by a few FP32 ulps.
        actual = torch.cat([pure(obs.unsqueeze(0)) for obs in probes])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    traced = torch.jit.trace(pure, probes[:1])
    traced = torch.jit.freeze(traced.eval(), optimize_numerics=False)
    torch.testing.assert_close(torch.cat([traced(obs.unsqueeze(0)) for obs in probes]), expected, atol=2e-6, rtol=2e-6)
    output.mkdir(parents=True)
    traced.save(str(output / "model.pt"))
    reloaded = torch.jit.load(str(output / "model.pt"), map_location="cpu")
    torch.testing.assert_close(torch.cat([reloaded(obs.unsqueeze(0)) for obs in probes]), expected, atol=2e-6, rtol=2e-6)
    metadata = {"format": "hydrogym-reference-actor/v1", "scope": "per-environment-reference-not-shared-submission",
                "paper_performance_status": "not-yet-matched", "algorithm": cfg["algorithm"],
                "seed": cfg["seed"], "training_step": artifact["training_step"],
                "observation_size": cfg["observations"], "action_size": cfg["actions"],
                "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
                "config": cfg, "physics": artifact["protocol"], "source": audit,
                "export_test": {"observations": len(probes), "max_abs_error": float((actual - expected).abs().max())}}
    (output / "config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if predictor is None:
        # environment/references/specialists in the repository, /workspace/references/specialists in
        # the image: both have the starter at ../../baseline/starter.
        predictor = Path(__file__).resolve().parents[2] / "baseline/starter/reference_predict.py"
    shutil.copyfile(predictor, output / "predict.py")
    (output / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nroot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec python "$root/predict.py"\n')
    (output / "README.md").write_text(
        "# Exported native reference actor\n\nThis is one per-environment baseline, not a shared multi-task candidate. "
        "Paper-performance matching has not been established. Training uses native HydroGym interactions; "
        "the exact configuration, seed, training step, source hashes and physical protocol are in config.json. "
        "Only this policy's learned weights and frozen preprocessing are used during inference. "
        "The model consumes observations, not environment IDs or action tables. "
        "No training directory, network, credentials or original author checkout is needed. "
        "Dependencies: Python and the declared PyTorch runtime.\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--predictor", type=Path)
    args = parser.parse_args()
    result = export(args.checkpoint, args.output, args.author_source, args.predictor)
    print(json.dumps({key: result[key] for key in ("algorithm", "training_step", "export_test", "model_sha256")}))


if __name__ == "__main__":
    main()

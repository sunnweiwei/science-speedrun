"""Measure supervised loss on the one materialized training example before/after fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .dataset import AlphaGenomeDataset
from .model import CompactModel, ModelConfig, count_parameters
from .train_eval import _autocast, _losses, _model_inputs, _to_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(model: CompactModel, sample: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.inference_mode(), _autocast(device):
        inputs, junction_target, junction_valid = _model_inputs(sample)
        losses = _losses(model(**inputs), sample, junction_target, junction_valid)
    return {name: float(value) for name, value in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    torch.manual_seed(args.seed)
    model = CompactModel(config).to(device)
    dataset = AlphaGenomeDataset(args.input, "train")
    model.set_junction_track_halves(dataset.junction_track_halves)
    sample = _to_device(dataset[0], device)
    initial = measure(model, sample, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    final = measure(model, sample, device)
    report = {
        "schema": "alphagenome-tiny-overfit/v1",
        "split": "train",
        "subsample": {"first_n": 1, "example_id": sample["example_id"]},
        "seed": args.seed,
        "epochs": checkpoint["epoch"],
        "global_steps": checkpoint["global_step"],
        "parameters": count_parameters(model),
        "input_length": config.input_length,
        "output_families": 11,
        "dataset_release_sha256": sha256(args.input / "release.json"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "initial_supervised_loss": initial,
        "final_supervised_loss": final,
        "total_loss_reduction_fraction": 1 - final["total"] / initial["total"],
        "diagnostic_scope": "same training example; teacher-forced splice coordinates",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "tiny_overfit_report", "initial": initial["total"],
                      "final": final["total"], "reduction": report["total_loss_reduction_fraction"]}))


if __name__ == "__main__":
    main()

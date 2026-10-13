#!/usr/bin/env python3
"""Minimal training/checkpoint flow; agents supply features, architecture, and loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import HORIZONS, build_training_loader, read_manifest
from model import build_model


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def compute_loss(outputs: dict, batch: dict, config: dict) -> torch.Tensor:
    """Implement the masked multi-horizon disease/death training objective here."""
    raise NotImplementedError(
        "Implement compute_loss with label masks, at-risk masks, and the chosen objective"
    )


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "metadata": metadata,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional smoke/debug limit; omit for real training")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    for section in ("model", "data", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.json requires an object-valued {section!r} section")
    manifest = read_manifest(args.input)
    loader, preprocessing_state = build_training_loader(args.input, config["data"])
    metadata = {
        "scored_diseases": list(manifest["scored_diseases"]),
        "horizons_days": list(HORIZONS),
        "preprocessing": preprocessing_state,
    }
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = build_model(config["model"], metadata).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    model.train()
    steps = 0
    for _epoch in range(int(config["training"]["epochs"])):
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(batch), batch, config["training"])
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("compute_loss must return one finite scalar tensor")
            loss.backward()
            optimizer.step()
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break
    if steps == 0:
        raise ValueError("Training loader produced no batches")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / config["training"]["checkpoint"]
    save_checkpoint(checkpoint, model, config, metadata)
    print(f"saved checkpoint after {steps} steps: {checkpoint}")


if __name__ == "__main__":
    main()

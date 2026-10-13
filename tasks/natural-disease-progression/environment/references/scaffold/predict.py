#!/usr/bin/env python3
"""Minimal raw-input inference driver for a submission-defined neural model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import (HORIZONS, build_preprocessor, collate_histories, iter_histories,
                  read_manifest, write_predictions)
from model import build_model


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    manifest = read_manifest(args.input)
    checkpoint_path = args.checkpoint or Path(__file__).resolve().with_name("checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model_state", "config", "metadata"} <= checkpoint.keys():
        raise ValueError("Checkpoint must contain model_state, config, and metadata")
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    if list(metadata.get("scored_diseases", ())) != list(manifest["scored_diseases"]):
        raise ValueError("Checkpoint disease vocabulary does not match the input manifest")
    if tuple(metadata.get("horizons_days", ())) != HORIZONS:
        raise ValueError("Checkpoint horizon grid does not match the task")
    preprocessor = build_preprocessor(config["data"], metadata.get("preprocessing"))
    model = build_model(config["model"], metadata)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    ids: list[str] = []
    disease_batches: list[np.ndarray] = []
    death_batches: list[np.ndarray] = []
    batch: list[dict] = []

    def flush() -> None:
        if not batch:
            return
        model_batch = move_to_device(collate_histories(batch, preprocessor), device)
        with torch.inference_mode():
            result = model(model_batch)
        if not isinstance(result, dict) or not {"disease_risk", "death_risk"} <= result.keys():
            raise ValueError("model.forward must return disease_risk and death_risk")
        disease_batches.append(result["disease_risk"].detach().cpu().numpy().astype(np.float32))
        death_batches.append(result["death_risk"].detach().cpu().numpy().astype(np.float32))
        batch.clear()

    for sample_id, history in iter_histories(args.input):
        ids.append(sample_id)
        batch.append(history)
        if len(batch) == args.batch_size:
            flush()
    flush()
    if not ids:
        raise ValueError("Input contains no samples")
    write_predictions(args.output, ids, np.concatenate(disease_batches),
                      np.concatenate(death_batches), len(manifest["scored_diseases"]))


if __name__ == "__main__":
    main()

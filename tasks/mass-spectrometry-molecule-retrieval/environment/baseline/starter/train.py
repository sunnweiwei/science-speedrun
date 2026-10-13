#!/usr/bin/env python3
"""Train and atomically publish the runnable Fourier-DeepSets starter."""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import LabeledSpectra, compact_json, sha256
from model import DeepSetsFingerprint
from publication import publish_directory


def cosine_loss(prediction, target):
    return 1 - F.cosine_similarity(prediction, target, dim=-1).mean()


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for spectra, targets in loader:
            spectra = spectra.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = cosine_loss(model(spectra), targets)
            total += float(loss) * spectra.shape[0]
            count += spectra.shape[0]
    return total / count


def atomic_torch_save(value, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--fingerprint-size", type=int, default=4096)
    parser.add_argument("--layers-per-mlp", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", type=Path, help="Optimizer checkpoints and logs, outside the submission (default: /workspace/research/<output name>-training)")
    parser.add_argument("--resume", type=Path, help="Resume last.pt at its completed epoch")
    parser.add_argument("--max-seconds", type=float, default=3600,
                        help="Cumulative fit budget, checked at epoch boundaries")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_seconds) <= 0 or args.workers < 0:
        raise ValueError("epochs must be positive")

    output = args.output_dir.resolve()
    work = (args.work_dir or Path("/workspace/research") / (output.name + "-training")).resolve()
    if work == output or work.is_relative_to(output):
        raise ValueError("keep optimizer checkpoints outside the submission")
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = LabeledSpectra(
        args.input_root / "train.jsonl", args.max_train_records, args.seed,
        fingerprint_size=args.fingerprint_size,
    )
    validation_data = LabeledSpectra(
        args.input_root / "validation.jsonl", args.max_validation_records, args.seed,
        fingerprint_size=args.fingerprint_size,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=loader_generator,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    config = {
        "hidden_size": args.hidden_size,
        "fingerprint_size": args.fingerprint_size,
        "layers_per_mlp": args.layers_per_mlp,
        "dropout": args.dropout,
        "max_peaks": 60,
        "mz_min": 10.0,
        "mz_max": 1000.0,
        "fingerprint_radius": 2,
    }
    device = torch.device(args.device)
    model = DeepSetsFingerprint(
        hidden_size=args.hidden_size,
        fingerprint_size=args.fingerprint_size,
        layers_per_mlp=args.layers_per_mlp,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    identity = {
        "schema": "massspecgym-retrieval-fit/v1", "initialization": "random", "config": config,
        "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "train_sha256": sha256(args.input_root / "train.jsonl"),
        "validation_sha256": sha256(args.input_root / "validation.jsonl"),
        "train_records": len(train_data), "validation_records": len(validation_data),
    }
    if (work / "recipe.json").exists() and (
        not args.resume or json.loads((work / "recipe.json").read_text()) != identity
    ):
        raise ValueError("use a fresh work directory or --resume with the same recipe")
    (work / "recipe.json").write_text(compact_json(identity) + "\n")
    best_loss, best_epoch, best_state = float("inf"), None, None
    first_epoch, prior = 1, 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("resume data/model/seed/batch configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        best_state, best_epoch, best_loss = saved["best_state"], saved["best_epoch"], saved["best_loss"]
        first_epoch, prior = saved["epoch"] + 1, saved["elapsed_seconds"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        loader_generator.set_state(saved["rng"]["loader"])
        if torch.cuda.is_available() and saved["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        if prior >= args.max_seconds:
            raise ValueError("resume has already exhausted the cumulative training budget")
    source = Path(__file__).resolve().parent

    def publish_best():
        summary = dict(identity, selected_epoch=best_epoch, best_validation_cosine_loss=best_loss,
                       elapsed_seconds=prior + time.monotonic() - started)
        def build(stage):
            for name in ("infer.sh", "predict.py", "model.py", "common.py", "train.py", "publication.py"):
                shutil.copyfile(source / name, stage / name)
            (stage / "infer.sh").chmod(0o755)
            checkpoint = stage / "model.pt"
            torch.save({"state_dict": best_state, "model_config": config}, checkpoint)
            (stage / "model_config.json").write_text(compact_json(config) + "\n")
            (stage / "training_summary.json").write_text(compact_json(summary) + "\n")
            (stage / "README.md").write_text(
                "# Trained Fourier-DeepSets retrieval model\n\n"
                "Initialized from scratch on the provided train fold; selected on validation loss. "
                "See training_summary.json for the data and recipe identity.\n")
            manifest = {
                "schema": "massspecgym-retrieval-model/v1",
                "checkpoint": {"path": "model.pt", "sha256": sha256(checkpoint), "format": "pytorch"},
                "model_config": "model_config.json",
                "training_provenance": {"data": "MassSpecGym v1.5 train fold", "pretrained_assets": []},
            }
            (stage / "model_manifest.json").write_text(compact_json(manifest) + "\n")
        publish_directory(output, build)

    if best_state is not None:
        publish_best()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for spectra, targets in train_loader:
            spectra = spectra.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = cosine_loss(model(spectra), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}; "
                    "check the feature precision and input values"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * spectra.shape[0]
            seen += spectra.shape[0]
        validation_loss = evaluate(model, validation_loader, device)
        if not np.isfinite(validation_loss):
            raise FloatingPointError(
                f"non-finite validation loss at epoch {epoch}"
            )
        print(compact_json({
            "epoch": epoch,
            "train_loss": running / seen,
            "validation_loss": validation_loss,
        }), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

            publish_best()
        elapsed = prior + time.monotonic() - started
        atomic_torch_save({
            "identity": identity, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_state": best_state, "best_epoch": best_epoch, "best_loss": best_loss,
            "elapsed_seconds": elapsed,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "loader": loader_generator.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        }, work / "last.pt")
        with (work / "metrics.jsonl").open("a") as stream:
            stream.write(compact_json({"epoch": epoch, "train_loss": running / seen,
                                       "validation_loss": validation_loss, "elapsed_seconds": elapsed}) + "\n")
        if elapsed >= args.max_seconds:
            break


if __name__ == "__main__":
    main()

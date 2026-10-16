"""LFMC full-width official UNet/head with effective-batch-preserving microbatches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from lfmc_batch import LFMCWindows, collate
from lfmc_model import LFMCModel, backward_microbatches


def digest_state(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, loader, device, limit=None, microbatch=2):
    model.eval()
    total_squared = 0.0
    total_absolute = 0.0
    count = 0
    windows = 0
    all_missing = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit is not None and step >= limit:
                break
            windows += len(batch["ids"])
            observed = torch.stack(
                [v.flatten(1).any(1) for v in batch["valid"].values()], dim=1
            ).any(1)
            all_missing += int((~observed).sum())
            for start in range(0, len(batch["ids"]), microbatch):
                end = start + microbatch
                pixels = {
                    k: v[start:end].to(device) for k, v in batch["pixels"].items()
                }
                valid = {k: v[start:end].to(device) for k, v in batch["valid"].items()}
                prediction = model(pixels, valid).outputs
                target = batch["target"][start:end].to(device)
                mask = batch["target_valid"][start:end].to(device)
                errors = (prediction - target)[mask].double()
                if not torch.isfinite(errors).all():
                    raise ValueError("Nonfinite validation prediction")
                total_squared += float(errors.square().sum())
                total_absolute += float(errors.abs().sum())
                count += len(errors)
    if not count:
        raise ValueError("No validation labels")
    return {
        "mse": total_squared / count,
        "mae": total_absolute / count,
        "valid_pixels": count,
        "windows": windows,
        "all_missing_windows": all_missing,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8123)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    if (
        os.environ.get("EARTH_SLURM_PARTITION")
        and os.environ.get("SLURM_JOB_PARTITION") != os.environ["EARTH_SLURM_PARTITION"]
    ):
        raise RuntimeError("Requires configured-gpu-partition")
    if a.output.exists():
        raise ValueError("Use a new output directory")
    index = json.loads(a.index.read_text())
    if index["counts"] != {"train": 31175, "val": 8262} or index["test_used"]:
        raise ValueError("Wrong data coverage")
    a.output.mkdir(parents=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda")
    model = LFMCModel(str(a.package)).to(device)
    initial_encoder = digest_state(model.encoder)
    initial_head = digest_state(model.decoder)
    train = LFMCWindows(index["rows"], "train", a.seed)
    val = LFMCWindows(index["rows"], "val", a.seed)
    train_loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(a.seed),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
        persistent_workers=False,
    )
    model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=2, cooldown=10, min_lr=0
    )
    epochs = 22 if a.pilot else 100
    limit = 1 if a.pilot else None
    settings = {
        "pilot": a.pilot,
        "seed": a.seed,
        "package": str(a.package),
        "encoder_file_sha256": model.encoder_sha256,
        "encoder_state_digest": initial_encoder,
        "head_state_digest": initial_head,
        "epochs": epochs,
        "unfreeze_epoch": 20,
        "global_batch": 32,
        "microbatch": a.microbatch,
        "batch_limit_per_epoch": limit,
        "counts": index["counts"],
        "validation_crop": "true label centered; declared deviation from official fixed random crop",
        "training_crop": "official uniform crop bounds with paired seed/epoch/window RNG",
        "test_used": False,
        "index_sha256": hashlib.sha256(a.index.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "train_lfmc.py",
                "lfmc_model.py",
                "lfmc_batch.py",
                "lfmc_window.py",
                "masked_spatial.py",
                "portable_encoder.py",
            ]
        },
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
    }
    (a.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    history = []
    best = float("inf")
    start_time = time.monotonic()
    for epoch in range(epochs):
        if epoch == 20:
            if digest_state(model.encoder) != initial_encoder:
                raise RuntimeError("Frozen encoder changed")
            model.encoder.requires_grad_(True)
            optimizer.add_param_group(
                {
                    "params": list(model.encoder.parameters()),
                    "lr": optimizer.param_groups[0]["lr"] / 10,
                }
            )
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            (a.output / "unfreeze-boundary.json").write_text(
                json.dumps({"epoch": 20, "frozen_weights_unchanged": True})
            )
        train.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        batches = 0
        valid_pixels = 0
        windows = 0
        for step, batch in enumerate(train_loader):
            if limit is not None and step >= limit:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, count = backward_microbatches(model, batch, device, a.microbatch)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            optimizer.step()
            # Match Lightning's epoch aggregation with batch_size=len(inputs).
            loss_sum += loss * len(batch["ids"])
            batches += 1
            valid_pixels += count
            windows += len(batch["ids"])
        validation = evaluate(model, val_loader, device, limit, a.microbatch)
        if not a.pilot and (windows != 31175 or validation["windows"] != 8262):
            raise ValueError("Incomplete epoch coverage")
        train_loss = loss_sum / windows
        scheduler.step(train_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_windows": windows,
            "train_valid_pixels": valid_pixels,
            "validation": validation,
            "encoder_trainable": epoch >= 20,
            "lr": [g["lr"] for g in optimizer.param_groups],
            "elapsed_seconds": time.monotonic() - start_time,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        history.append(row)
        (a.output / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if validation["mse"] < best:
            best = validation["mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "settings": settings,
                },
                a.output / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "settings": settings,
            },
            a.output / "last.pt",
        )
    final = digest_state(model.encoder)
    if final == initial_encoder:
        raise RuntimeError("Encoder never updated after unfreezing")
    (a.output / "completion.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "encoder_changed": True,
                "final_encoder_state_digest": final,
                "best_validation_mse": best,
                "test_used": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

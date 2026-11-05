#!/usr/bin/env python3
"""Train and publish the minimal atom-histogram starter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch.utils.data import ConcatDataset, DataLoader

from common import sha256
from data import ELEMENTS, PairedDataset
from model import ModelConfig, StarterModel


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    total = count = 0
    for batch in loader:
        if len(batch["molecule"]) < 2:
            continue
        loss = model.loss(batch["molecule"].to(device), batch["pocket"].to(device))
        total += float(loss) * len(batch["molecule"])
        count += len(batch["molecule"])
    if not count:
        raise ValueError("validation needs at least one batch with two examples")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("starter-model"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-pretrain-records", type=int)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    parser.add_argument("--without-profsa", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("epochs must be positive, batch size >=2, and workers nonnegative")
    for name in ("max_pretrain_records", "max_train_records",
                 "max_validation_records"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    torch.manual_seed(args.seed)
    root = args.input_root.resolve()
    train_parts = []
    if not args.without_profsa:
        train_parts.append(PairedDataset(
            root / "processed/profsa_pretrain_500k/data.lmdb", args.max_pretrain_records
        ))
    train_parts.append(PairedDataset(
        root / "processed/biolip_train/data.lmdb", args.max_train_records
    ))
    training = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    validation = PairedDataset(
        root / "processed/biolip_dev/data.lmdb", args.max_validation_records
    )
    if len(training) < args.batch_size:
        raise ValueError("training split must contain at least one complete batch")
    if len(validation) < 2:
        raise ValueError("validation split needs at least two examples")
    device = torch.device(args.device)
    config = ModelConfig(len(ELEMENTS), args.hidden_dim, args.embedding_dim)
    model = StarterModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)
    best_loss, best_state, best_epoch = float("inf"), None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch["molecule"].to(device), batch["pocket"].to(device))
            loss.backward()
            optimizer.step()
        current = validation_loss(model, valid_loader, device)
        print(json.dumps({"epoch": epoch, "validation_loss": current}), flush=True)
        if current < best_loss:
            best_loss, best_epoch = current, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    source = Path(__file__).resolve().parent
    try:
        for name in ("model.py", "data.py", "common.py", "predict.py", "infer.sh",
                     "requirements.txt"):
            shutil.copyfile(source / name, stage / name)
        (stage / "infer.sh").chmod(0o755)
        checkpoint = stage / "model.pt"
        torch.save({"state_dict": best_state, "model_config": config.to_dict()}, checkpoint)
        (stage / "model_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        summary = {"best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "train_records": len(training), "validation_records": len(validation),
                   "parameter_count": sum(value.numel() for value in model.parameters())}
        (stage / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "schema": "target-based-drug-screening-model/v1",
            "method": "atom-histogram-starter",
            "checkpoint": {"path": "model.pt", "sha256": sha256(checkpoint)},
            "model_config": "model_config.json",
            "parameter_count": summary["parameter_count"],
            "training_summary": "training_summary.json",
        }
        (stage / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

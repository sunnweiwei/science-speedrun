#!/usr/bin/env python3
"""DDP trainer for compact reference folding architectures."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from reference.data import FoldingReferenceDataset, collate_reference_batch
from reference.losses import folding_losses, prepare_noisy_batch
from reference.models import ReferenceFoldingConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.stack([metrics[name].detach().float() for name in names])
    if world_size > 1:
        dist.all_reduce(values)
        values /= world_size
    return {
        name: float(value) for name, value in zip(names, values, strict=True)
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    state = {
        "schema_version": "reference-folding-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    dataset: FoldingReferenceDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    rank: int,
    world_size: int,
    max_batches: int,
    seed: int,
) -> dict[str, float]:
    dataset.set_epoch(0)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_reference_batch,
    )
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    batches = 0
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    for batch in loader:
        if batches >= max_batches:
            break
        batch = prepare_noisy_batch(move_to_device(batch, device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = folding_losses(model(batch), batch)
        for name, value in losses.items():
            totals[name] = totals.get(name, torch.zeros_like(value.detach().float())) + value.detach().float()
        batches += 1
    if not batches:
        raise RuntimeError("validation produced no batches")
    count = torch.tensor(float(batches), device=device)
    if world_size > 1:
        dist.all_reduce(count)
        for value in totals.values():
            dist.all_reduce(value)
    return {name: float(value / count) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    training = config["training"]
    if args.steps is not None:
        training = dict(training)
        training["steps"] = args.steps
        config = dict(config)
        config["training"] = training
    model_config = ReferenceFoldingConfig.from_dict(config["model"])
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_float32_matmul_precision("high")
    seed = int(training["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    train_dataset = FoldingReferenceDataset(
        args.train,
        crop_size=int(training["crop_size"]),
        training=True,
        seed=seed,
    )
    validation_dataset = FoldingReferenceDataset(
        args.validation,
        crop_size=int(training["crop_size"]),
        training=False,
        seed=seed,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    global_step = 0
    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["step"])
        epoch = int(checkpoint["epoch"])
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("a") if rank == 0 else None
    metadata = {
        "schema_version": "reference-folding-run-v1",
        "architecture": model_config.architecture,
        "parameters": raw_model.parameter_count(),
        "world_size": world_size,
        "train_manifest": str(args.train),
        "train_manifest_sha256": file_sha256(args.train),
        "validation_manifest": str(args.validation),
        "validation_manifest_sha256": file_sha256(args.validation),
        "config": config,
        "resume": str(args.resume) if args.resume else None,
    }
    if rank == 0:
        (args.output_dir / "run.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metadata, sort_keys=True), flush=True)

    target_steps = int(training["steps"])
    accumulate = int(training.get("gradient_accumulation", 1))
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    while global_step < target_steps:
        train_dataset.set_epoch(epoch)
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        ) if world_size > 1 else None
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=int(training["workers"]),
            pin_memory=True,
            collate_fn=collate_reference_batch,
        )
        for micro_step, batch in enumerate(loader, 1):
            batch = prepare_noisy_batch(move_to_device(batch, device))
            sync_step = micro_step % accumulate == 0
            context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not sync_step
                else nullcontext()
            )
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = folding_losses(model(batch), batch)
                    scaled_loss = losses["loss"] / accumulate
                scaled_loss.backward()
            if not sync_step:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            reduced = reduce_metrics(losses, world_size)
            if world_size > 1:
                gradient_norm_tensor = gradient_norm.detach().float().to(device)
                dist.all_reduce(gradient_norm_tensor)
                gradient_norm_tensor /= world_size
                gradient_norm = gradient_norm_tensor
            record = {
                "type": "train",
                "step": global_step,
                "epoch": epoch,
                **reduced,
                "gradient_norm": float(gradient_norm),
                "seconds": time.monotonic() - started,
            }
            if rank == 0 and (
                global_step == 1 or global_step % int(training["log_every"]) == 0
            ):
                assert metrics_handle is not None
                metrics_handle.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_handle.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            if global_step % int(training["validate_every"]) == 0 or global_step == target_steps:
                validation = validate(
                    model,
                    validation_dataset,
                    int(training["batch_size"]),
                    int(training["workers"]),
                    device,
                    rank,
                    world_size,
                    int(training["validation_batches"]),
                    seed + 100000,
                )
                model.train()
                if rank == 0:
                    validation_record = {
                        "type": "validation",
                        "step": global_step,
                        **validation,
                    }
                    assert metrics_handle is not None
                    metrics_handle.write(json.dumps(validation_record, sort_keys=True) + "\n")
                    metrics_handle.flush()
                    print(json.dumps(validation_record, sort_keys=True), flush=True)
            if global_step % int(training["checkpoint_every"]) == 0 or global_step == target_steps:
                if rank == 0:
                    save_checkpoint(
                        args.output_dir / f"checkpoint-step-{global_step:07d}.pt",
                        raw_model,
                        optimizer,
                        global_step,
                        epoch,
                        config,
                    )
                if world_size > 1:
                    dist.barrier()
            if global_step >= target_steps:
                break
        epoch += 1

    if rank == 0:
        assert metrics_handle is not None
        metrics_handle.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

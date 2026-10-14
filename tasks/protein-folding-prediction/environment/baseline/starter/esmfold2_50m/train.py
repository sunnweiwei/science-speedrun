#!/usr/bin/env python3
"""DDP training for the from-scratch approximately-50M ESMFold2 starter."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from .adapter import prepare_training_example
from .architecture import build_model, persistent_state_dict, training_forward
from .config import EXPECTED_PARAMETER_RANGE, MODEL_NAME_50M
from .recipe import (
    ProjectRecipeConfig,
    cosine_learning_rate,
    parameter_count,
    synchronize_gradients,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, default=Path("/opt/esmfold2-assets"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-atoms", type=int, default=4096)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-loops", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def folding_state(model) -> dict[str, Any]:
    return persistent_state_dict(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )


def save_checkpoint(path, model, optimizer, ema, step, metadata) -> None:
    import torch
    import torch.distributed as dist

    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": "foldbench-esmfold2-final-hidden-checkpoint/v1",
                "model": folding_state(model),
                "ema": {name: value.detach().cpu() for name, value in ema.items()},
                "optimizer": optimizer.state_dict(),
                "step": step,
                "metadata": metadata,
            },
            temporary,
        )
        temporary.replace(path)
    dist.barrier()


def record_skipped_example(output_dir, rank, step, record, error):
    """Append one line per example the model itself could not process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "step": step,
        "rank": rank,
        "example_id": record["example_id"],
        "error_type": type(error).__name__,
        "error": str(error).splitlines()[0][:300] if str(error) else "",
    }
    with (output_dir / "skipped_examples.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"skipped {record['example_id']} on rank {rank} at step {step}: "
          f"{type(error).__name__}: {entry['error']}", flush=True)


def next_example(records, start, max_tokens, max_atoms, builder):
    """Deterministically skip records the upstream all-atom adapter cannot represent."""

    errors = []
    for offset in range(len(records)):
        record = records[(start + offset) % len(records)]
        if int(record.get("token_count", 0)) > max_tokens:
            continue
        if not record.get("sequences"):
            # 71 pool entries have no polymer chain at all (ligand/ion-only
            # assemblies); they carry nothing to fold, and the one-token case
            # (pdb:2z41) crashes the released pair embedder.
            continue
        try:
            prepared = prepare_training_example(record, builder)
            features = prepared[0]
            actual_tokens = int(features["token_index"].shape[-1])
            actual_atoms = int(features["ref_pos"].shape[-2])
            if actual_tokens > max_tokens or actual_atoms > max_atoms:
                continue
            return record, prepared, offset
        except Exception as error:  # auditable adapter/capability rejection
            if len(errors) < 3:
                errors.append(f"{record['example_id']}: {type(error).__name__}: {error}")
    raise RuntimeError(f"no representable training example; first errors={errors}")


def main() -> None:
    args = arguments()
    if min(args.max_tokens, args.max_atoms, args.checkpoint_interval, args.log_interval) < 1:
        raise ValueError("token/atom/checkpoint/log values must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema decay must be between zero and one")

    import numpy as np
    import torch
    import torch.distributed as dist
    from esm.models.esmfold2 import ESMFold2InputBuilder

    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    records = read_records(args.train)
    if not records:
        raise ValueError("empty training index")
    random.Random(args.seed).shuffle(records)
    shard = records[rank::world_size]
    default_steps = math.ceil(len(records) / world_size)
    max_steps = args.max_steps if args.max_steps is not None else default_steps
    if max_steps < 1:
        raise ValueError("max steps must be positive")

    model = build_model()
    count = parameter_count(model)
    if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
        raise AssertionError(f"{MODEL_NAME_50M} has {count:,} parameters")
    model.to(device)
    # Biohub's fused SwiGLU cache is inference-only and is regenerated from
    # canonical weights. Training must use the differentiable reference path
    # so strict checkpoints contain only the learned model state.
    model.set_kernel_backend(None)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    recipe = replace(ProjectRecipeConfig(), total_steps=max_steps)
    optimizer = torch.optim.AdamW(
        trainable, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    start_step = 0
    resume = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(persistent_state_dict(resume["model"]), strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = int(resume["step"])

    ema = (
        {name: value.to(device) for name, value in resume["ema"].items()}
        if rank == 0 and resume is not None
        else {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if rank == 0
        else {}
    )
    builder = ESMFold2InputBuilder(ccd_cache=args.ccd_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    cursor = start_step
    model.train()
    for step_index in range(start_step, max_steps):
        record, (features, target, target_mask), skipped = next_example(
            shard, cursor % len(shard), args.max_tokens, args.max_atoms, builder
        )
        cursor += skipped + 1
        features = {
            key: value.to(device, non_blocking=True)
            for key, value in features.items()
        }
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = training_forward(
                    model,
                    features,
                    target,
                    target_mask,
                    recipe=recipe,
                    num_loops=args.num_loops,
                )
            losses["loss"].backward()
        except Exception as error:  # noqa: BLE001 - one example must not end a run
            # The adapter accepted this record but the released graph rejected
            # it (or ran out of memory). Every rank must still take part in the
            # gradient all-reduce below, so this rank contributes zero gradient
            # for the step and records the example instead of aborting the run.
            record_skipped_example(args.output_dir, rank, step_index + 1, record, error)
            optimizer.zero_grad(set_to_none=True)
            for parameter in trainable:
                parameter.grad = torch.zeros_like(parameter)
            losses = {name: torch.zeros((), device=device)
                      for name in ("loss", "diffusion_loss", "distogram_loss")}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        synchronize_gradients(trainable, world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, recipe.gradient_clip_norm)
        finite = torch.tensor(
            [float(torch.isfinite(losses["loss"]) and torch.isfinite(grad_norm))],
            device=device,
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"non-finite step {step_index + 1}")
        lr = cosine_learning_rate(step_index, recipe)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if rank == 0:
            named = dict(model.named_parameters())
            for name, value in ema.items():
                value.mul_(args.ema_decay).add_(named[name].detach(), alpha=1 - args.ema_decay)

        completed = step_index + 1
        metrics = {
            "step": completed,
            "example_id": record["example_id"],
            "loss": float(losses["loss"].detach()),
            "diffusion_loss": float(losses["diffusion_loss"].detach()),
            "distogram_loss": float(losses["distogram_loss"].detach()),
            "gradient_norm": float(grad_norm),
            "learning_rate": lr,
            "adapter_skips": skipped,
            "tokens": int(features["token_index"].shape[-1]),
            "atoms": int(features["ref_pos"].shape[-2]),
        }
        if rank == 0 and (completed == 1 or completed % args.log_interval == 0):
            with log_path.open("a") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
        if completed == 1 or completed % args.checkpoint_interval == 0 or completed == max_steps:
            save_checkpoint(
                args.output_dir / "checkpoints" / f"step-{completed:07d}.pt",
                model,
                optimizer,
                ema,
                completed,
                {
                    "model": MODEL_NAME_50M,
                    "parameter_count": count,
                    "seed": args.seed,
                    "world_size": world_size,
                    "max_tokens": args.max_tokens,
                    "max_atoms": args.max_atoms,
                    "activation_dtype": "bfloat16",
                    "training_pool": str(args.train),
                    "fixed_feature": "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1",
                    "pretrained_parameters": False,
                },
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

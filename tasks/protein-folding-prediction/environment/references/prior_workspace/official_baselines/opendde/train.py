#!/usr/bin/env python3
"""Train the scaled released OpenDDE graph on an audited common-data view.

OpenDDE publishes inference architecture and weights, not its trainer.  This
runner uses only released network modules and the transparent project-owned
``foldingtrain_edm_v1`` optimizer/loss Recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from agent_pipeline.contract import load_json, sha256_file
from official_baselines.opendde.config import EXPECTED_PARAMETER_RANGE, configure_ccd
from official_baselines.project_recipe import (
    ProjectRecipeConfig,
    cosine_learning_rate,
    parameter_count,
    synchronize_gradients,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--valid-manifest", type=Path)
    parser.add_argument("--native-view-audit", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--ccd-dir", type=Path, default=Path("/opendde_data/common"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--n-cycle", type=int, default=1)
    parser.add_argument("--validation-examples", type=int, default=28)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    # Architecture/synthetic-graph audit mode.
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-input", type=Path, default=Path("examples/opendde_smoke.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit(model) -> dict[str, Any]:
    count = parameter_count(model)
    if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
        raise AssertionError(f"OpenDDE scaled parameter count {count:,} is outside budget")
    return {
        "baseline": "opendde_structural_50m_v1",
        "architecture_source": "official_opendde_modules",
        "parameters": count,
        "structural_token_expansion": True,
        "structural_refiner": True,
        "training_recipe": ProjectRecipeConfig().metadata(),
    }


def synthetic_smoke(model, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from opendde.data.inference.infer_dataloader import InferenceDataset

    from official_baselines.opendde.trainable_forward import training_forward

    configure_ccd(args.ccd_dir)
    model.configs.data.ccd_components_file = str(args.ccd_dir / "components.cif")
    model.configs.data.ccd_components_rdkit_mol_file = str(
        args.ccd_dir / "components.cif.rdkit_mol.pkl"
    )
    model.configs.input_json_path = str(args.smoke_input.resolve())
    model.configs.use_msa = False
    model.configs.use_template = False
    dataset = InferenceDataset(model.configs)
    data, _atom_array, error = dataset[0]
    if error:
        raise RuntimeError(error)
    device = torch.device(args.device)
    features = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data["input_feature_dict"].items()
    }
    model.to(device).train()
    model.confidence_head.requires_grad_(False)
    target = features["ref_pos"].float().clone()
    target_mask = features["ref_mask"].bool()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4, weight_decay=1.0e-2)
    optimizer.zero_grad(set_to_none=True)
    losses = training_forward(model, features, target, target_mask, n_cycle=1)
    losses["loss"].backward()
    grad_parameters = sum(parameter.numel() for parameter in parameters if parameter.grad is not None)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return {
        "smoke_only": True,
        "native_training_example": False,
        "loss": float(losses["loss"].detach()),
        "diffusion_loss": float(losses["diffusion_loss"].detach()),
        "distogram_loss": float(losses["distogram_loss"].detach()),
        "parameters_with_gradient": grad_parameters,
        "gradient_norm": float(grad_norm),
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rank_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rank_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def set_ema_weights(model, ema: dict[str, Any]) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    parameters = dict(model.named_parameters())
    for name, value in ema.items():
        previous[name] = parameters[name].detach().cpu().clone()
        parameters[name].data.copy_(value.to(parameters[name].device))
    return previous


def restore_weights(model, state: dict[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    ema: dict[str, Any],
    step: int,
    unique_examples: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> None:
    import torch
    import torch.distributed as dist

    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(rank_rng_state(), gathered, dst=0)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": "opendde-foldingtrain-checkpoint/v1",
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "ema": ema,
                "step": step,
                "global_exposures": step * world_size,
                "unique_native_examples": unique_examples,
                "rank_rng_states": gathered,
                "seed": args.seed,
                "n_cycle": args.n_cycle,
            },
            temporary,
        )
        temporary.replace(path)
    dist.barrier()


def validate_audit(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    if args.native_view_audit is None:
        raise ValueError("--native-view-audit is required for production training")
    native = load_json(args.native_view_audit)
    if native["model_id"] != "opendde_structural_50m":
        raise ValueError("native view audit belongs to another model")
    if native["accepted_examples"] != len(records):
        raise ValueError("native view accepted count does not match train manifest")
    if native["native_manifest_sha256"] != sha256_file(args.train_manifest):
        raise ValueError("train manifest does not match native view audit")
    if native["selection_policy"] != "capability_only":
        raise ValueError("native view must use capability_only selection")
    return native


def production_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_manifest is None or args.output_dir is None:
        raise ValueError("--train-manifest and --output-dir are required")
    if args.n_cycle < 1 or args.checkpoint_interval < 1 or args.log_interval < 1:
        raise ValueError("cycle/checkpoint/log values must be positive")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be between zero and one")

    import numpy as np
    import torch
    import torch.distributed as dist

    from official_baselines.opendde.data_adapter import load_example
    from official_baselines.opendde.trainable_forward import build_model, training_forward

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    configure_ccd(args.ccd_dir)

    records = read_records(args.train_manifest)
    native_audit = validate_audit(args, records)
    ordered = list(records)
    random.Random(args.seed).shuffle(ordered)
    shard = ordered[rank::world_size]
    default_steps = math.ceil(len(ordered) / world_size)
    max_steps = args.max_steps if args.max_steps is not None else default_steps
    if max_steps < 1 or max_steps > default_steps:
        raise ValueError(f"--max-steps must be in [1, {default_steps}]")

    torch.manual_seed(args.seed)
    model = build_model().to(device)
    report = audit(model)
    model.confidence_head.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    recipe = replace(ProjectRecipeConfig(), total_steps=max_steps)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )
    start_step = 0
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["step"])
        if resume_payload["seed"] != args.seed:
            raise ValueError("resume seed mismatch")
        restore_rank_rng(resume_payload["rank_rng_states"][rank])
    else:
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
    ema = (
        {
            name: value.to(device)
            for name, value in resume_payload["ema"].items()
        }
        if rank == 0 and resume_payload is not None
        else {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if rank == 0
        else {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_metrics.jsonl"
    last_metrics: dict[str, float] = {}
    model.train()
    for step_index in range(start_step, max_steps):
        record = shard[step_index % len(shard)]
        features, target, target_mask = load_example(record, args.data_root)
        features = {key: value.to(device, non_blocking=True) for key, value in features.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16",
        ):
            losses = training_forward(
                model,
                features,
                target.to(device, non_blocking=True),
                target_mask.to(device, non_blocking=True),
                recipe=recipe,
                n_cycle=args.n_cycle,
                chunk_size=args.chunk_size,
            )
        losses["loss"].backward()
        synchronize_gradients(trainable, world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, recipe.gradient_clip_norm)
        finite = torch.tensor(
            [float(torch.isfinite(losses["loss"]).item() and torch.isfinite(grad_norm).item())],
            device=device,
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"non-finite loss/gradient at step {step_index + 1}")
        lr = cosine_learning_rate(step_index, recipe)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if rank == 0:
            parameters = dict(model.named_parameters())
            for name in ema:
                ema[name].mul_(args.ema_decay).add_(
                    parameters[name].detach(), alpha=1.0 - args.ema_decay
                )
        completed = step_index + 1
        last_metrics = {
            "loss": float(losses["loss"].detach()),
            "diffusion_loss": float(losses["diffusion_loss"].detach()),
            "distogram_loss": float(losses["distogram_loss"].detach()),
            "gradient_norm": float(grad_norm),
            "learning_rate": lr,
        }
        if rank == 0 and (completed == 1 or completed % args.log_interval == 0):
            row = {"step": completed, "global_exposures": completed * world_size, **last_metrics}
            with log_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_interval == 0 or completed == max_steps:
            save_checkpoint(
                args.output_dir / "checkpoints" / f"step-{completed:07d}.pt",
                model,
                optimizer,
                ema,
                completed,
                len(records),
                args,
                rank,
                world_size,
            )

    validation: dict[str, Any] | None = None
    if rank == 0 and args.valid_manifest is not None and args.validation_examples > 0:
        valid = read_records(args.valid_manifest)[: args.validation_examples]
        previous = set_ema_weights(model, ema)
        model.eval()
        totals = {"loss": 0.0, "diffusion_loss": 0.0, "distogram_loss": 0.0}
        for record in valid:
            features, target, target_mask = load_example(record, args.data_root)
            features = {key: value.to(device) for key, value in features.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.precision == "bf16",
            ):
                losses = training_forward(
                    model,
                    features,
                    target.to(device),
                    target_mask.to(device),
                    recipe=recipe,
                    n_cycle=args.n_cycle,
                    chunk_size=args.chunk_size,
                )
            for name in totals:
                totals[name] += float(losses[name])
            del losses
        validation = {
            "examples": len(valid),
            "ema": True,
            **{name: value / max(1, len(valid)) for name, value in totals.items()},
        }
        restore_weights(model, previous)
        (args.output_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
    dist.barrier()
    result = {
        **report,
        "native_view": native_audit,
        "world_size": world_size,
        "optimizer_steps": max_steps,
        "global_exposures": max_steps * world_size,
        "unique_native_examples": len(records),
        "padding_exposures": max_steps * world_size - len(records),
        "execution": {
            "precision": args.precision,
            "attention_chunk_size": args.chunk_size,
            "activation_checkpoint_blocks": 1,
            "fine_grained_diffusion_checkpoint": True,
        },
        "completed": max_steps == default_steps,
        "last_metrics": last_metrics if rank == 0 else None,
        "validation": validation,
    }
    if rank == 0:
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()
    return result


def main() -> None:
    args = arguments()
    if args.train_manifest is not None:
        production_train(args)
        return
    from official_baselines.opendde.trainable_forward import build_model

    model = build_model()
    report = audit(model)
    if args.smoke:
        report["smoke"] = synthetic_smoke(model, args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

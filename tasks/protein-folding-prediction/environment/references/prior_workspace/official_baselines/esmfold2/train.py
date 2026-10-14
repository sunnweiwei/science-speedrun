#!/usr/bin/env python3
"""Train the scaled official ESMFold2 graph on its audited native view.

Biohub did not publish its trainer.  The network graph is the pinned released
implementation; optimizer, schedule and EDM/distogram objective are the
explicit project-owned ``foldingtrain_edm_v1`` Recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from agent_pipeline.contract import FOLDBENCH_TRACKS, load_json, sha256_file
from official_baselines.esmfold2.config import EXPECTED_PARAMETER_RANGE
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hf-home", type=Path, default=Path("/data/hf"))
    parser.add_argument(
        "--ccd-dir",
        type=Path,
        default=Path(
            "/data/hf/models--biohub--ESMFold2/snapshots/"
            "1ebf0e3481a5184eb6171d40615c79e384b48796"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--num-loops", type=int, default=0)
    parser.add_argument("--validation-examples", type=int, default=28)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260802)
    # Retain the lightweight architecture/synthetic graph audit.
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", default="ACDEFGHIK")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit(model) -> dict[str, Any]:
    count = parameter_count(model)
    if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
        raise AssertionError(f"ESMFold2 scaled parameter count {count:,} is outside budget")
    return {
        "baseline": "esmfold2_fast_50m_v1",
        "architecture_source": "official_biohub_modules",
        "parameters": count,
        "frozen_embedding_parameters_in_budget": False,
        "training_recipe": ProjectRecipeConfig().metadata(),
    }


def synthetic_smoke(model, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    from official_baselines.esmfold2.trainable_forward import training_forward

    device = torch.device(args.device)
    model.to(device).train()
    model.confidence_head.requires_grad_(False)
    features = {key: value.to(device) for key, value in prepare_protein_features(args.sequence).items()}
    batch, tokens = features["res_type"].shape
    features["lm_hidden_states"] = torch.randn(
        batch,
        tokens,
        model.config.lm_num_layers + 1,
        model.config.lm_d_model,
        device=device,
        dtype=torch.bfloat16,
    )
    target = features["ref_pos"].float().clone()
    target_mask = features["atom_attention_mask"].bool()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4, weight_decay=1.0e-2)
    optimizer.zero_grad(set_to_none=True)
    losses = training_forward(model, features, target, target_mask, num_loops=0)
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


def deterministic_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return ordered


def select_validation_records(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select a small deterministic, track-covered, uncropped health check.

    Antibody–antigen is first in the round-robin order so the campaign's 28-row
    health check contains four examples from that priority track and three from
    every other FoldBench track.  Small complexes are preferred to keep this
    internal loss diagnostic separate from the full external FoldBench run.
    """

    if count < 1:
        return []
    track_order = (
        "interface_antibody_antigen",
        *(track for track in FOLDBENCH_TRACKS if track != "interface_antibody_antigen"),
    )
    grouped = {
        track: sorted(
            (record for record in records if record.get("task") == track),
            key=lambda record: (int(record["token_count"]), record["example_id"]),
        )
        for track in track_order
    }
    positions = {track: 0 for track in track_order}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        added = False
        for track in track_order:
            position = positions[track]
            if position >= len(grouped[track]):
                continue
            selected.append(grouped[track][position])
            positions[track] += 1
            added = True
            if len(selected) == count:
                break
        if not added:
            raise ValueError(
                f"validation manifest has only {len(selected)} FoldBench rows; "
                f"requested {count}"
            )
    return selected


def crop_start(record: dict[str, Any], crop_size: int, seed: int) -> int:
    excess = int(record["length"]) - crop_size
    if excess <= 0:
        return 0
    digest = hashlib.sha256(f"{seed}:{record['example_id']}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (excess + 1)


def folding_state(model) -> dict[str, Any]:
    from official_baselines.esmfold2.trainable_forward import persistent_state_dict

    return persistent_state_dict({
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("_esmc.")
    })


def set_ema_weights(model, ema: dict[str, Any]) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    parameters = dict(model.named_parameters())
    for name, value in ema.items():
        previous[name] = parameters[name].detach().cpu().clone()
        parameters[name].data.copy_(value.to(parameters[name].device))
    return previous


def restore_weights(model, previous: dict[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in previous.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    ema: dict[str, Any],
    step: int,
    records: int,
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
                "schema_version": "esmfold2-foldingtrain-checkpoint/v1",
                "model": folding_state(model),
                "optimizer": optimizer.state_dict(),
                "ema": ema,
                "step": step,
                "global_exposures": step * world_size,
                "unique_native_examples": records,
                "rank_rng_states": gathered,
                "seed": args.seed,
                "crop_size": args.crop_size,
                "num_loops": args.num_loops,
            },
            temporary,
        )
        temporary.replace(path)
    dist.barrier()


def validate_audit(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    if args.native_view_audit is None:
        raise ValueError("--native-view-audit is required for production training")
    native = load_json(args.native_view_audit)
    if native["model_id"] != "esmfold2_fast_50m":
        raise ValueError("native view audit belongs to a different model")
    if native["accepted_examples"] != len(records):
        raise ValueError("native view accepted count does not match train manifest")
    if native["native_manifest_sha256"] != sha256_file(args.train_manifest):
        raise ValueError("train manifest hash does not match native view audit")
    if native["selection_policy"] != "capability_only":
        raise ValueError("native view must use capability_only selection")
    return native


def production_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_manifest is None or args.output_dir is None:
        raise ValueError("--train-manifest and --output-dir are required")
    if args.crop_size < 1 or args.checkpoint_interval < 1 or args.log_interval < 1:
        raise ValueError("crop/checkpoint/log intervals must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be between zero and one")

    import numpy as np
    import torch
    import torch.distributed as dist
    from official_baselines.esmfold2.data_adapter import load_cropped_example
    from official_baselines.esmfold2.trainable_forward import (
        build_model,
        initialize_frozen_lm_projection,
        persistent_state_dict,
        training_forward,
    )
    from esm.models.esmfold2 import ESMFold2InputBuilder

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

    records = read_records(args.train_manifest)
    native_audit = validate_audit(args, records)
    ordered = deterministic_order(records, args.seed)
    shard = ordered[rank::world_size]
    default_steps = math.ceil(len(ordered) / world_size)
    max_steps = args.max_steps if args.max_steps is not None else default_steps
    if max_steps < 1 or max_steps > default_steps:
        raise ValueError(f"--max-steps must be in [1, {default_steps}] for one-pass training")

    # Identical initialization is required because gradients are averaged
    # manually; ESMC is attached only after trainable parameters/optimizer exist.
    torch.manual_seed(args.seed)
    model = build_model()
    projection = initialize_frozen_lm_projection(model, args.hf_home)
    model = model.to(device)
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
        model.load_state_dict(persistent_state_dict(resume_payload["model"]), strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["step"])
        if resume_payload["seed"] != args.seed:
            raise ValueError("resume seed mismatch")

    model.set_kernel_backend("fused")
    input_builder = ESMFold2InputBuilder(ccd_cache=args.ccd_dir)
    if resume_payload is not None:
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
    last_losses: dict[str, float] = {}
    model.train()
    for step_index in range(start_step, max_steps):
        # The last few ranks deterministically wrap once when N is not divisible
        # by world size.  The exact padding count is recorded in the checkpoint.
        record = shard[step_index % len(shard)]
        try:
            features, target, target_mask = load_cropped_example(
                record,
                args.data_root,
                crop_start=crop_start(record, args.crop_size, args.seed),
                crop_size=args.crop_size,
                input_builder=input_builder,
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to load native example {record['example_id']}"
            ) from error
        features = {key: value.to(device, non_blocking=True) for key, value in features.items()}
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        losses = training_forward(
            model,
            features,
            target,
            target_mask,
            recipe=recipe,
            num_loops=args.num_loops,
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
        last_losses = {
            "loss": float(losses["loss"].detach()),
            "diffusion_loss": float(losses["diffusion_loss"].detach()),
            "distogram_loss": float(losses["distogram_loss"].detach()),
            "gradient_norm": float(grad_norm),
            "learning_rate": lr,
        }
        if rank == 0 and (completed == 1 or completed % args.log_interval == 0):
            row = {"step": completed, "global_exposures": completed * world_size, **last_losses}
            with log_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        # The first checkpoint gives an external anytime evaluator a valid
        # publication even when the Controller stops a short training run.
        if completed == 1 or completed % args.checkpoint_interval == 0 or completed == max_steps:
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

    if rank == 0 and not last_losses and log_path.is_file():
        logged = read_records(log_path)
        if logged:
            last_losses = {
                name: float(value)
                for name, value in logged[-1].items()
                if name not in {"step", "global_exposures"}
            }

    validation: dict[str, Any] | None = None
    if rank == 0 and args.valid_manifest is not None and args.validation_examples > 0:
        valid = select_validation_records(
            read_records(args.valid_manifest), args.validation_examples
        )
        previous = set_ema_weights(model, ema)
        model.eval()
        totals = {"loss": 0.0, "diffusion_loss": 0.0, "distogram_loss": 0.0}
        for record in valid:
            features, target, target_mask = load_cropped_example(
                record,
                args.data_root,
                crop_start=0,
                crop_size=max(args.crop_size, int(record["token_count"])),
                input_builder=input_builder,
            )
            features = {key: value.to(device) for key, value in features.items()}
            losses = training_forward(
                model,
                features,
                target.to(device),
                target_mask.to(device),
                recipe=recipe,
                num_loops=args.num_loops,
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
        "precomputed_esmc": projection,
        "native_view": native_audit,
        "world_size": world_size,
        "optimizer_steps": max_steps,
        "global_exposures": max_steps * world_size,
        "unique_native_examples": len(records),
        "padding_exposures": max(0, max_steps * world_size - len(records)),
        "unexposed_native_examples": max(0, len(records) - max_steps * world_size),
        "completed": max_steps == default_steps,
        "last_metrics": last_losses if rank == 0 else None,
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
    from official_baselines.esmfold2.trainable_forward import build_model

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

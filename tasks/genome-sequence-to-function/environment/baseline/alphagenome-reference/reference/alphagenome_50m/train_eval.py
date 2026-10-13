"""Train and stream-evaluate the <=50M compact AlphaGenome baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import (
    AlphaGenomeDataset,
    DENSE_FAMILIES,
    single_example_collate,
)
from .model import (
    ALL_FAMILIES,
    CompactModel,
    ModelConfig,
    count_parameters,
)


DEFAULT_CONFIG = Path(__file__).with_name("alphagenome_50m.yaml")


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _autocast(device: torch.device):
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _distributed() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) under torchrun; (0, 0, 1) otherwise."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    return int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0")), world


def _loader(dataset, *, workers: int, shuffle: bool, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=single_example_collate,
    )


def _to_device(value, device: torch.device):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _match_coordinate_queries(
    scoring: dict[str, Tensor], positions: Tensor, track_halves: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Match trusted coordinate queries to typed position axes like the grader."""
    device = positions.device
    tracks = scoring["track"]
    halves = track_halves[tracks]
    queries = torch.zeros((len(tracks), 3), dtype=torch.int64, device=device)
    queries[:, 2] = tracks
    valid = scoring["valid"].clone()
    missing = torch.ones(len(tracks), dtype=torch.bool, device=device)
    position_rows = positions.detach().cpu().numpy()
    halves_cpu = halves.detach().cpu().numpy()
    donors_cpu = scoring["donor_position"].detach().cpu().numpy()
    acceptors_cpu = scoring["acceptor_position"].detach().cpu().numpy()
    valid_indices = np.flatnonzero(valid.detach().cpu().numpy())
    lookups = [
        {int(coordinate): index for index, coordinate in enumerate(axis) if coordinate >= 0}
        for axis in position_rows
    ]
    for index in valid_indices:
        half = int(halves_cpu[index])
        row = lookups[2 * half].get(int(donors_cpu[index]))
        column = lookups[2 * half + 1].get(int(acceptors_cpu[index]))
        if row is not None and column is not None:
            queries[index, 0] = row
            queries[index, 1] = column
            missing[index] = False
    return queries.unsqueeze(0), valid, missing


def _model_inputs(sample: dict) -> tuple[dict, Tensor, Tensor]:
    queries, query_valid, missing = _match_coordinate_queries(
        sample["junction_scoring"],
        sample["site_positions"],
        sample["junction_track_halves"],
    )
    if (query_valid & missing).any():
        raise ValueError("trusted scoring coordinates do not match target positions")
    junction_target = torch.asinh(
        sample["junction_scoring"]["target_value"]
    ).unsqueeze(0)
    return {
        "dna": sample["dna"].unsqueeze(0),
        "usage_positions": sample["usage_positions"].unsqueeze(0),
        "junction_site_positions": sample["site_positions"].unsqueeze(0),
        "junction_queries": queries,
    }, junction_target, query_valid


def _losses(outputs: dict[str, Tensor], sample: dict, junction_target: Tensor,
            junction_valid: Tensor) -> dict[str, Tensor]:
    losses = {}
    for family in DENSE_FAMILIES:
        target = torch.asinh(sample["dense"][family]["values"]).unsqueeze(0)
        mask = sample["dense"][family]["mask"].unsqueeze(0)
        residual = torch.where(mask, outputs[family] - target, 0.0)
        losses[family] = residual.square().sum() / mask.sum().clamp_min(1)
    losses["splice_sites"] = F.binary_cross_entropy_with_logits(
        outputs["splice_sites"], sample["splice_sites"].unsqueeze(0)
    )
    usage_target = torch.asinh(sample["splice_usage"]["values"]).unsqueeze(0)
    usage_mask = sample["splice_usage"]["mask"].unsqueeze(0)
    usage_residual = torch.where(usage_mask, outputs["splice_usage"] - usage_target, 0.0)
    losses["splice_usage"] = usage_residual.square().sum() / usage_mask.sum().clamp_min(1)
    junction_residual = outputs["junctions"][0] - junction_target[0]
    losses["junctions"] = (
        junction_residual[junction_valid].square().mean()
        if junction_valid.any() else junction_residual.sum() * 0
    )
    contact_target = torch.asinh(sample["contact_maps"]).unsqueeze(0)
    contact_mask = sample["contact_map_mask"].unsqueeze(0)
    contact_residual = torch.where(
        contact_mask, outputs["contact_maps"] - contact_target, 0.0
    )
    losses["contact_maps"] = contact_residual.square().sum() / contact_mask.sum().clamp_min(1)
    losses["total"] = torch.stack([losses[name] for name in ALL_FAMILIES]).mean()
    return losses


def _validation_inference(
    model: CompactModel, sample: dict
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
    """Use the same predicted-position path as a formal v3 submission."""
    trunk = model._trunk(sample["dna"].unsqueeze(0))
    dense_features = model._dense_features(trunk)
    outputs = {
        family: model.dense_heads[family](dense_features).float()
        for family in DENSE_FAMILIES
    }
    all_sites = model._splice_site_logits(trunk)
    target_start = (model.config.input_length - model.config.target_length) // 2
    outputs["splice_sites"] = all_sites[:, target_start:target_start + model.config.target_length]
    usage_bins = model.splice_usage_head(dense_features).float()
    usage_indices = torch.div(sample["usage_positions"], model.config.dense_bin_size,
                              rounding_mode="floor")
    outputs["splice_usage"] = usage_bins[:, usage_indices]
    predicted_positions = model.select_junction_positions(all_sites)

    matched, valid, missing = _match_coordinate_queries(
        sample["junction_scoring"],
        predicted_positions[0],
        sample["junction_track_halves"],
    )
    junction_target = torch.asinh(
        sample["junction_scoring"]["target_value"]
    ).unsqueeze(0)
    junction_prediction = torch.zeros(len(valid), device=valid.device)
    if (~missing & valid).any():
        selected = ~missing & valid
        junction_prediction[selected] = model._junction_predictions(
            trunk, predicted_positions, matched[:, selected]
        )[0]
    outputs["junctions"] = junction_prediction.unsqueeze(0)

    contact = trunk.reshape(
        1, model.config.contact_bins, -1, model.config.hidden_dim
    ).mean(dim=2)
    q, k = model.contact_q(contact), model.contact_k(contact)
    row, column = model.contact_row, model.contact_column
    outputs["contact_maps"] = (0.5 * (q[:, row] * k[:, column] + q[:, column] * k[:, row])).float()
    return outputs, junction_target, valid, missing


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_position(payload: dict, *, examples: int, epochs: int) -> tuple[int, int]:
    epoch = int(payload["epoch"])
    step = int(payload["global_step"])
    kind = payload.get("checkpoint_kind")
    if kind not in (None, "epoch") or step != epoch * examples:
        raise ValueError("only a completed-epoch checkpoint can resume training")
    if payload.get("examples", examples) != examples or payload.get("schedule_epochs", epochs) != epochs:
        raise ValueError("checkpoint data size or training schedule differs from this run")
    return epoch, step


def train(args: argparse.Namespace) -> None:
    # Multi-GPU with `torchrun --nproc-per-node N train_eval.py train ...`: one example per GPU per
    # step (data parallel), rank 0 writes checkpoints and logs. Model size is bounded by the
    # training budget, not by a parameter cap.
    rank, local_rank, world = _distributed()
    config = ModelConfig.from_yaml(args.config)
    device = torch.device("cuda", local_rank) if world > 1 else _device(args.device)
    if world > 1:
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    model = CompactModel(config).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    full_dataset = AlphaGenomeDataset(args.input, "train")
    model.set_junction_track_halves(full_dataset.junction_track_halves)
    size = min(len(full_dataset), args.max_examples or len(full_dataset))
    dataset = Subset(full_dataset, range(size))
    start_epoch, global_step = 0, 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("config") != config.to_dict():
            raise ValueError("checkpoint configuration does not match --config")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload.get("scaler", {}))
        start_epoch, global_step = _resume_position(payload, examples=size, epochs=args.epochs)
    per_rank = math.ceil(size / world)
    steps = max(1, per_rank * args.epochs)
    module = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    sampler = (torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world, rank=rank,
                                                               shuffle=True, seed=args.seed)
               if world > 1 else None)
    stop_after_epoch = args.stop_after_epoch or args.epochs
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if rank == 0:
        print(json.dumps({"event": "train_start", "parameters": parameters, "examples": size,
                          "device": str(device), "world_size": world}), flush=True)

    for epoch in range(start_epoch, stop_after_epoch):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = _loader(dataset, workers=args.workers, shuffle=True, sampler=sampler)
        for sample in loader:
            sample = _to_device(sample, device)
            inputs, junction_target, junction_valid = _model_inputs(sample)
            progress = min((global_step + 1) / steps, 1.0)
            learning_rate = args.min_learning_rate + 0.5 * (
                args.learning_rate - args.min_learning_rate
            ) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                predictions = model(**inputs)
                losses = _losses(predictions, sample, junction_target, junction_valid)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if rank == 0 and (global_step == 1 or global_step % args.log_every == 0):
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": float(losses["total"].detach()),
                            "gradient_norm": float(gradient_norm),
                            "learning_rate": learning_rate,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if rank == 0 and global_step % args.save_every == 0:
                _atomic_save(
                    {
                        "schema": "alphagenome-50m-real/v3",
                        "model": module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "checkpoint_kind": "progress",
                        "examples": size,
                        "schedule_epochs": args.epochs,
                    },
                    output / "progress.pt",
                )
        if world > 1:
            torch.distributed.barrier()
        if rank != 0:
            continue
        _atomic_save(
            {
                "schema": "alphagenome-50m-real/v3",
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "checkpoint_kind": "epoch",
                "examples": size,
                "schedule_epochs": args.epochs,
            },
            output / "last.pt",
        )
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return
    checkpoint = output / "last.pt"
    manifest = {
        "schema": "alphagenome-50m-real/v3",
        "checkpoint": {"path": checkpoint.name, "sha256": _sha256(checkpoint)},
        "parameters": parameters,
        "config": config.to_dict(),
        "output_families": list(ALL_FAMILIES),
        "epochs": args.epochs,
        "completed_epochs": stop_after_epoch,
        "global_step": global_step,
    }
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "train_complete", **manifest}), flush=True)


class PerTrackMoments:
    def __init__(self, tracks: int):
        self.n = np.zeros(tracks, dtype=np.int64)
        self.sum_x = np.zeros(tracks, dtype=np.float64)
        self.sum_y = np.zeros(tracks, dtype=np.float64)
        self.sum_x2 = np.zeros(tracks, dtype=np.float64)
        self.sum_y2 = np.zeros(tracks, dtype=np.float64)
        self.sum_xy = np.zeros(tracks, dtype=np.float64)

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        x = prediction.detach().double().cpu().numpy().reshape(-1, prediction.shape[-1])
        y = target.detach().double().cpu().numpy().reshape(-1, target.shape[-1])
        valid = np.ones_like(y, dtype=bool) if mask is None else mask.detach().cpu().numpy().reshape(y.shape).astype(bool)
        self.n += valid.sum(axis=0)
        self.sum_x += np.where(valid, x, 0).sum(axis=0, dtype=np.float64)
        self.sum_y += np.where(valid, y, 0).sum(axis=0, dtype=np.float64)
        self.sum_x2 += np.where(valid, x * x, 0).sum(axis=0, dtype=np.float64)
        self.sum_y2 += np.where(valid, y * y, 0).sum(axis=0, dtype=np.float64)
        self.sum_xy += np.where(valid, x * y, 0).sum(axis=0, dtype=np.float64)

    def update_indexed(self, prediction: Tensor, target: Tensor, track: Tensor) -> None:
        x = prediction.detach().double().cpu().numpy().reshape(-1)
        y = target.detach().double().cpu().numpy().reshape(-1)
        indices = track.detach().cpu().numpy().reshape(-1).astype(np.int64)
        self.n += np.bincount(indices, minlength=len(self.n))
        self.sum_x += np.bincount(indices, weights=x, minlength=len(self.n))
        self.sum_y += np.bincount(indices, weights=y, minlength=len(self.n))
        self.sum_x2 += np.bincount(indices, weights=x * x, minlength=len(self.n))
        self.sum_y2 += np.bincount(indices, weights=y * y, minlength=len(self.n))
        self.sum_xy += np.bincount(indices, weights=x * y, minlength=len(self.n))

    def report(self) -> dict:
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denominator = np.sqrt(
            np.maximum(self.n * self.sum_x2 - self.sum_x**2, 0)
            * np.maximum(self.n * self.sum_y2 - self.sum_y**2, 0)
        )
        target_ss = np.maximum(self.n * self.sum_y2 - self.sum_y**2, 0)
        scorable = (self.n >= 2) & (target_ss > 0)
        pearson = np.full_like(numerator, np.nan)
        nonconstant = scorable & (denominator > 0)
        pearson[nonconstant] = numerator[nonconstant] / denominator[nonconstant]
        pearson[scorable & ~nonconstant] = 0.0
        return {
            "mean_pearson": float(pearson[scorable].mean()) if scorable.any() else 0.0,
            "valid_tracks": int(scorable.sum()),
            "per_track_pearson": [float(value) if np.isfinite(value) else None for value in pearson],
            "values_per_track": self.n.tolist(),
        }


def evaluate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["config"])
    model = CompactModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    full_dataset = AlphaGenomeDataset(args.input, args.split)
    model.set_junction_track_halves(full_dataset.junction_track_halves)
    size = min(len(full_dataset), args.max_examples or len(full_dataset))
    dataset = Subset(full_dataset, range(size))
    metrics = {
        **{family: PerTrackMoments(config_value) for family, config_value in {
            "atac": 167, "dnase": 305, "procap": 12, "cage": 546,
            "rna_seq": 667, "chip_tf": 1617, "chip_histone": 1116,
        }.items()},
        "splice_sites": PerTrackMoments(4),
        "splice_usage": PerTrackMoments(config.splice_usage_tracks),
        "junctions": PerTrackMoments(config.junction_tracks),
        "contact_maps": PerTrackMoments(config.contact_tracks),
    }
    with torch.inference_mode():
        for index, sample in enumerate(_loader(dataset, workers=args.workers, shuffle=False)):
            sample = _to_device(sample, device)
            outputs, junction_target, junction_valid, junction_missing = (
                _validation_inference(model, sample)
            )
            # Match predict.py: FP32 inference is serialized to FP16 Zarr values.
            outputs = {name: value.half().float() for name, value in outputs.items()}
            for family in DENSE_FAMILIES:
                metrics[family].update(
                    outputs[family][0],
                    torch.asinh(sample["dense"][family]["values"]),
                    sample["dense"][family]["mask"],
                )
            metrics["splice_sites"].update(outputs["splice_sites"][0], sample["splice_sites"])
            metrics["splice_usage"].update(
                outputs["splice_usage"][0],
                torch.asinh(sample["splice_usage"]["values"]),
                sample["splice_usage"]["mask"],
            )
            junction_track = sample["junction_scoring"]["track"]
            metrics["junctions"].update_indexed(
                outputs["junctions"][0][junction_valid],
                junction_target[0][junction_valid],
                junction_track[junction_valid],
            )
            metrics["contact_maps"].update(
                outputs["contact_maps"][0],
                torch.asinh(sample["contact_maps"]),
                sample["contact_map_mask"],
            )
            if index == 0 or (index + 1) % args.log_every == 0 or index + 1 == size:
                print(json.dumps({"event": "evaluate_progress", "examples": index + 1, "total": size}), flush=True)
    families = {name: accumulator.report() for name, accumulator in metrics.items()}
    valid_family_scores = [item["mean_pearson"] for item in families.values()]
    report = {
        "schema": "alphagenome-50m-evaluation/v3",
        "split": args.split,
        "examples": size,
        "metric_space": (
            "asinh continuous targets and binary splice-site targets; "
            "junction Pearson uses hidden coordinate queries and predicted endpoints; "
            "FP32 inference rounded to FP16 prediction values"
        ),
        "families": families,
        "overall_mean_family_pearson": float(np.mean(valid_family_scores)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "evaluate_complete", "overall_mean_family_pearson": report["overall_mean_family_pearson"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", required=True, type=Path)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--workers", type=int, default=2)
    common.add_argument("--max-examples", type=int, default=0)
    common.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    common.add_argument("--log-every", type=int, default=25)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--stop-after-epoch", type=int)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    train_parser.add_argument("--save-every", type=int, default=250)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--resume", type=Path)
    train_parser.set_defaults(run=train)
    evaluate_parser = subparsers.add_parser("evaluate", parents=[common])
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    evaluate_parser.add_argument("--report", required=True, type=Path)
    evaluate_parser.set_defaults(run=evaluate)
    args = parser.parse_args()
    if args.workers < 0 or args.max_examples < 0:
        parser.error("workers/max-examples must be nonnegative")
    if args.command == "train" and (
        args.epochs < 1
        or (args.stop_after_epoch is not None
            and not 1 <= args.stop_after_epoch <= args.epochs)
    ):
        parser.error("epochs must be positive and stop-after-epoch must be within the schedule")
    args.run(args)


if __name__ == "__main__":
    main()

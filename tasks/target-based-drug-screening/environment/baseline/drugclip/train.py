#!/usr/bin/env python3
"""Paper-scale dual Uni-Mol DrugCLIP reference on the fixed materialized data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common import atomic_torch_save, compact_json, publish_directory, sha256
from data import POCKET_CROP_POLICY, PairedDataset, collate_pairs, move_features
from model import PaperConfig, PaperDrugCLIP, parameter_count
from training import PolynomialSchedule, optimizer_updates
from validation import aggregate_metrics, retrieval_batch_metrics, selection_improves

MOLECULE_SHA256 = "da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9"
MOLECULE_DICTIONARY_SHA256 = "94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127"
POCKET_DICTIONARY_SHA256 = "5775af31a33d78c5f5f2555c4506debbd2d47c6b29bf8a889237220b34b2c7d5"
DEFAULT_STAGE = {
    "profsa": {"epochs": 100, "learning_rate": 1e-4, "gpus": 8},
    "finetune": {"epochs": 200, "learning_rate": 1e-3, "gpus": 4},
}
METRIC_KEYS = ("loss_sum", "bedroc_sum", "auc_sum", "top1_sum", "top3_sum", "queries")


def data_identity(root: Path) -> dict:
    manifest = root / "processed/manifest.json"
    complete = root / ".complete.json"
    if not manifest.is_file() or not complete.is_file():
        raise FileNotFoundError(f"incomplete materialization: {root}")
    return {"processed_manifest_sha256": sha256(manifest),
            "completion_sha256": sha256(complete)}


def pretrained_identity(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "drugclip-pretrained-assets/v2":
        raise ValueError("expected dual-Uni-Mol pretrained asset manifest v2")
    for name, expected in (("molecule", MOLECULE_SHA256),
                           ("molecule_dictionary", MOLECULE_DICTIONARY_SHA256),
                           ("pocket_dictionary", POCKET_DICTIONARY_SHA256)):
        item = manifest[name]
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise FileNotFoundError(f"missing pretrained asset: {name}")
        if item["sha256"] != expected or path.stat().st_size != item["size"] or sha256(path) != expected:
            raise ValueError(f"pretrained {name} size or SHA256 mismatch")
    return manifest


def require_compatible_profsa_crop(identity: dict, seed: int) -> None:
    crop = identity.get("pocket_crop", {})
    if crop.get("policy") != POCKET_CROP_POLICY or crop.get("seed") != seed:
        raise ValueError("ProFSA checkpoint uses an incompatible pocket crop policy or seed")


def distributed_setup(device_arg: str) -> tuple[torch.device, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if device_arg.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device(device_arg)
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(device_arg)
    return device, rank, world


def make_training_loader(dataset, args, rank: int, world: int):
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate_pairs,
                        pin_memory=args.device.startswith("cuda"), drop_last=True)
    if len(loader) < 1:
        raise ValueError("stage has fewer records than world size times per-GPU batch")
    return loader, sampler


def validation_batches(records: int, batch_size: int, rank: int,
                       world: int) -> list[list[int]]:
    # Each rank sees distinct rows; the final singleton joins the preceding
    # batch so that it still has a retrieval candidate and is never discarded.
    indices = list(range(rank, records, world))
    batches = [indices[i:i + batch_size]
               for i in range(0, len(indices), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    if len(batches) == 1 and len(batches[0]) == 1:
        raise ValueError("each validation rank requires at least two records")
    return batches


def make_validation_loader(dataset, args, rank: int, world: int):
    batches = validation_batches(len(dataset), args.validation_batch_size,
                                 rank, world)
    return DataLoader(dataset, batch_sampler=batches, num_workers=args.workers,
                      collate_fn=collate_pairs,
                      pin_memory=args.device.startswith("cuda"))


def training_loss(logits, batch):
    return PaperDrugCLIP.contrastive_loss(
        logits, batch["molecule_ids"], batch["pocket_ids"])


@torch.no_grad()
def evaluate(model, loader, device, world: int) -> dict:
    model.eval()
    totals = {key: 0.0 for key in METRIC_KEYS}
    for batch in loader:
        logits = model(move_features(batch["molecule"], device),
                       move_features(batch["pocket"], device))
        result = retrieval_batch_metrics(
            logits, batch["molecule_ids"], batch["pocket_ids"])
        for key in METRIC_KEYS:
            totals[key] += result[key]
    if world > 1:
        vector = torch.tensor([totals[key] for key in METRIC_KEYS],
                              device=device, dtype=torch.float64)
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        totals = dict(zip(METRIC_KEYS, vector.cpu().tolist(), strict=True))
    return aggregate_metrics(totals)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def gather_rng(world: int) -> list[dict]:
    local = rng_state()
    if world == 1:
        return [local]
    collected = [None] * world
    dist.all_gather_object(collected, local)
    return collected


STOP_CODES = {"continue": 0, "next_stage": 1, "time_budget": 2,
              "patience": 3, "max_epochs": 4}
STOP_NAMES = {value: key for key, value in STOP_CODES.items()}


def stop_decision(*, stage: str, epoch: int, max_epochs: int,
                  another_stage: bool, elapsed: float,
                  max_seconds: float | None, patience_counter: int,
                  patience: int) -> str:
    """Decide at an epoch boundary; a completed final stage wins over time."""
    if epoch >= max_epochs:
        if not another_stage:
            return "max_epochs"
        if max_seconds is not None and elapsed >= max_seconds:
            return "time_budget"
        return "next_stage"
    if stage == "finetune" and patience_counter >= patience:
        return "patience"
    if max_seconds is not None and elapsed >= max_seconds:
        return "time_budget"
    return "continue"


def broadcast_stop(decision: str | None, elapsed: float, *,
                   device: torch.device, world: int) -> tuple[str, float]:
    if world == 1:
        assert decision is not None
        return decision, elapsed
    rank = dist.get_rank()
    payload = torch.tensor(
        [STOP_CODES[decision] if rank == 0 else 0, elapsed if rank == 0 else 0],
        dtype=torch.float64, device=device,
    )
    dist.broadcast(payload, src=0)
    return STOP_NAMES[int(payload[0].item())], float(payload[1].item())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("profsa", "finetune", "both"), required=True)
    p.add_argument("--input-root", type=Path, default=Path("/data"))
    p.add_argument("--pretrained-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("drugclip-paper-model"))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--profsa-checkpoint", type=Path,
                   help="required for a separate fine-tuning process")
    p.add_argument("--resume", type=Path)
    p.add_argument("--profsa-epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=200)
    p.add_argument("--profsa-learning-rate", type=float, default=1e-4)
    p.add_argument("--finetune-learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=48,
                   help="per GPU; negatives stay local to this batch")
    p.add_argument("--validation-batch-size", type=int, default=128)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--polynomial-power", type=float, default=1.0)
    p.add_argument("--end-learning-rate", type=float, default=0.0)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-molecule-atoms", type=int, default=256)
    p.add_argument("--max-pocket-atoms", type=int, default=256)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seconds", type=float,
                   help="optional stop at an epoch boundary with a resumable checkpoint")
    p.add_argument("--max-pretrain-records", type=int)
    p.add_argument("--max-train-records", type=int)
    p.add_argument("--max-validation-records", type=int)
    args = p.parse_args()
    if min(args.profsa_epochs, args.finetune_epochs, args.batch_size,
           args.validation_batch_size, args.gradient_accumulation,
           args.profsa_learning_rate, args.finetune_learning_rate,
           args.clip_norm, args.patience) <= 0 or args.workers < 0:
        p.error("training parameters must be positive")
    if args.batch_size < 2 or args.validation_batch_size < 2:
        p.error("contrastive batches require at least two candidates")
    if not 0 <= args.warmup_ratio < 1 or args.polynomial_power <= 0:
        p.error("invalid polynomial schedule")
    if args.max_seconds is not None and args.max_seconds <= 0:
        p.error("--max-seconds must be positive")
    for name in ("max_pretrain_records", "max_train_records", "max_validation_records"):
        value = getattr(args, name)
        if value is not None and value < 2:
            p.error(f"--{name.replace('_', '-')} must be at least two")
    if args.stage == "finetune" and not args.profsa_checkpoint:
        p.error("separate fine-tuning and resume require --profsa-checkpoint")
    if args.stage != "finetune" and args.profsa_checkpoint:
        p.error("ProFSA initializes its pocket tower randomly")
    return args


def main() -> None:
    args = parse_args()
    device, rank, world = distributed_setup(args.device)
    root = args.input_root.resolve()
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    if work == output or work.is_relative_to(output):
        raise ValueError("optimizer checkpoints must be outside the published model")
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    assets_dir = (args.pretrained_dir or root / "pretrained").resolve()
    assets = pretrained_identity(assets_dir)
    data = data_identity(root)
    config = PaperConfig(dropout=args.dropout,
                         max_molecule_atoms=args.max_molecule_atoms,
                         max_pocket_atoms=args.max_pocket_atoms,
                         gradient_checkpointing=args.gradient_checkpointing)
    config.validate()
    options = dict(max_molecule_atoms=config.max_molecule_atoms,
                   max_pocket_atoms=config.max_pocket_atoms, seed=args.seed,
                   unimol_molecule=True, unimol_pocket=True)
    profsa = (PairedDataset(root / "processed/profsa_pretrain_500k/data.lmdb",
                            max_records=args.max_pretrain_records, **options)
              if args.stage in ("profsa", "both") else None)
    biolip = (PairedDataset(root / "processed/biolip_train/data.lmdb",
                            max_records=args.max_train_records, **options)
              if args.stage in ("finetune", "both") else None)
    dev = (PairedDataset(root / "processed/biolip_dev/data.lmdb",
                         max_records=args.max_validation_records, **options)
           if biolip is not None else None)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    model = PaperDrugCLIP(config)
    initial = model.molecule_encoder.load_official_checkpoint(
        assets_dir / assets["molecule"]["path"], MOLECULE_SHA256)
    source_checkpoint = None
    source_formal = False
    source_epoch = source_updates = source_samples = 0
    source_identity = None
    if args.profsa_checkpoint:
        source = args.profsa_checkpoint.resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if payload.get("stage") != "profsa" or not payload.get("complete"):
            raise ValueError("fine-tuning requires a completed ProFSA checkpoint")
        require_compatible_profsa_crop(payload["identity"], args.seed)
        model.load_state_dict(payload["model"], strict=True)
        source_checkpoint = {"path": str(source), "sha256": sha256(source),
                             "completed_epochs": payload["epoch"],
                             "completed_updates": payload.get("updates"),
                             "samples": payload.get("samples")}
        source_formal = bool(payload["identity"].get("formal_recipe"))
        source_identity = payload["identity"]
        source_epoch = int(payload["epoch"])
        source_updates = int(payload.get("updates") or 0)
        source_samples = int(payload.get("samples") or
                             payload["identity"].get("profsa", {}).get("samples", 0))
    model.to(device)
    count = parameter_count(model)
    def schedule_plan(stage_data, epochs):
        if stage_data is None:
            return None
        stage_loader, _ = make_training_loader(stage_data, args, rank, world)
        per_epoch = optimizer_updates(len(stage_loader), args.gradient_accumulation)
        total = epochs * per_epoch
        return {"updates_per_epoch": per_epoch, "total_updates": total,
                "warmup_updates": int(args.warmup_ratio * total)}

    if profsa is not None:
        profsa_plan = schedule_plan(profsa, args.profsa_epochs)
    elif source_identity is not None:
        profsa_plan = source_identity.get("profsa", {}).get("schedule")
    else:
        profsa_plan = None
    finetune_plan = schedule_plan(biolip, args.finetune_epochs) if biolip else None
    formal = (args.batch_size == 48 and args.validation_batch_size == 128
              and args.gradient_accumulation == 1 and args.warmup_ratio == 0.06
              and args.polynomial_power == 1.0 and args.end_learning_rate == 0.0
              and args.profsa_epochs == 100 and args.finetune_epochs == 200
              and args.profsa_learning_rate == 1e-4
              and args.finetune_learning_rate == 1e-3
              and args.seed == 1 and args.clip_norm == 1.0
              and args.dropout == 0.1 and args.max_molecule_atoms == 256
              and args.max_pocket_atoms == 256
              and args.gradient_checkpointing and args.fp16
              and args.patience == 20
              and args.max_pretrain_records is None and args.max_train_records is None
              and args.max_validation_records is None and args.max_seconds is None
              and world == (8 if args.stage == "profsa" else 4)
              and args.stage != "both"
              and (args.stage != "finetune" or source_formal))
    identity = {
        "schema": "target-based-drug-screening-fit/v3", "method": "paper-dual-unimol",
        "dataset": data, "pretrained_assets": assets, "pretrained_load": initial,
        "profsa_source_checkpoint": source_checkpoint,
        "model_config": config.to_dict(), "parameter_count": count,
        "stage": args.stage, "world_size": world, "per_gpu_batch": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_batch": world * args.batch_size * args.gradient_accumulation,
        "contrastive_candidates_per_query": args.batch_size,
        "cross_gpu_negatives": False,
        "validation_batch_size": args.validation_batch_size,
        "validation_batch_sizes_by_rank": (
            [[len(batch) for batch in validation_batches(
                len(dev), args.validation_batch_size, r, world)]
             for r in range(world)] if dev is not None else []),
        "validation_protocol": "fixed BioLiP dev order; epoch-0 conformer/crop; local batch pocket-to-ligand BEDROC alpha=80.5; mean over queries",
        "pocket_crop": {"policy": POCKET_CROP_POLICY, "seed": args.seed,
                        "seed_mixing": "unicore_numpy_seed(seed, epoch, index)",
                        "dev_epoch": 0, "test_epoch": 0,
                        "test_max_atoms": 511},
        "profsa": {"max_epochs": args.profsa_epochs,
                   "samples": len(profsa) if profsa else source_samples,
                   "learning_rate": args.profsa_learning_rate,
                   "schedule": profsa_plan},
        "finetune": {"max_epochs": args.finetune_epochs,
                     "samples": len(biolip) if biolip else 0,
                     "learning_rate": args.finetune_learning_rate,
                     "schedule": finetune_plan,
                     "dev_samples": len(dev) if dev else 0},
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0},
        "scheduler": {"name": "polynomial_decay", "power": args.polynomial_power,
                      "end_learning_rate": args.end_learning_rate,
                      "warmup_ratio": args.warmup_ratio},
        "selection": {"metric": "valid_bedroc", "maximize": True,
                      "patience": args.patience, "alpha": 80.5},
        "seed": args.seed, "clip_norm": args.clip_norm, "fp16": args.fp16,
        "fp16_initial_scale": 4.0, "fp16_growth_interval": 256,
        "formal_recipe": formal, "max_seconds": args.max_seconds,
    }
    recipe = work / "recipe.json"
    if rank == 0:
        if recipe.exists() and (not args.resume or json.loads(recipe.read_text()) != identity):
            raise ValueError("use a fresh work directory or resume with the same recipe")
        recipe.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    if world > 1:
        dist.barrier()
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda",
                                  init_scale=4.0, growth_interval=256)
    saved = None
    best_bedroc = float("-inf")
    best_epoch = None
    patience_counter = 0
    elapsed_before = 0.0
    global_updates = source_updates
    completed = {"profsa": source_epoch, "finetune": 0}
    completed_updates = {"profsa": source_updates, "finetune": 0}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("resume recipe differs from current run")
        model.load_state_dict(saved["model"], strict=True)
        scaler.load_state_dict(saved["scaler"])
        best_bedroc = saved["best_bedroc"]
        best_epoch = saved["best_epoch"]
        patience_counter = saved["patience_counter"]
        elapsed_before = saved["elapsed_seconds"]
        global_updates = saved["global_updates"]
        completed = saved["completed_epochs"]
        completed_updates = saved["completed_updates"]
        restore_rng(saved["rng_by_rank"][rank])
    started = time.monotonic()
    source_dir = Path(__file__).resolve().parent
    stop_reason = "max_epochs"
    stages = [("profsa", profsa, args.profsa_epochs, args.profsa_learning_rate),
              ("finetune", biolip, args.finetune_epochs, args.finetune_learning_rate)]
    for stage_name, dataset, max_epochs, base_lr in stages:
        if dataset is None:
            continue
        if completed[stage_name] >= max_epochs:
            continue
        model.set_stage(stage_name)
        train_loader, sampler = make_training_loader(dataset, args, rank, world)
        plan = profsa_plan if stage_name == "profsa" else finetune_plan
        updates_per_epoch = optimizer_updates(len(train_loader), args.gradient_accumulation)
        if updates_per_epoch != plan["updates_per_epoch"]:
            raise ValueError("stage DataLoader changed after recipe creation")
        total_updates = plan["total_updates"]
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=base_lr, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0)
        scheduler = PolynomialSchedule(
            optimizer, base_lr=base_lr, total_updates=total_updates,
            warmup_ratio=args.warmup_ratio, power=args.polynomial_power,
            end_lr=args.end_learning_rate)
        if saved and saved["stage"] == stage_name:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
        if world > 1:
            wrapped = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False)
        else:
            wrapped = model
        dev_loader = (make_validation_loader(dev, args, rank, world)
                      if stage_name == "finetune" else None)
        for stage_epoch in range(completed[stage_name] + 1, max_epochs + 1):
            dataset.set_epoch(stage_epoch)
            sampler.set_epoch(stage_epoch)
            if dev is not None:
                dev.set_epoch(0)
            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = examples = 0.0
            for batch_index, batch in enumerate(train_loader, 1):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=args.fp16 and device.type == "cuda"):
                    logits = wrapped(move_features(batch["molecule"], device),
                                     move_features(batch["pocket"], device))
                    loss = training_loss(logits, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {stage_name} loss")
                window = min(args.gradient_accumulation,
                             len(train_loader) - ((batch_index - 1) // args.gradient_accumulation)
                             * args.gradient_accumulation)
                scaler.scale(loss / window).backward()
                loss_total += float(loss.detach()) * len(batch["molecule_ids"])
                examples += len(batch["molecule_ids"])
                if batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= scale_before:
                        scheduler.step()
                        global_updates += 1
                        completed_updates[stage_name] += 1
            if world > 1:
                vector = torch.tensor([loss_total, examples], device=device,
                                      dtype=torch.float64)
                dist.all_reduce(vector, op=dist.ReduceOp.SUM)
                loss_total, examples = vector.tolist()
            metrics = {"stage": stage_name, "epoch": stage_epoch,
                       "global_updates": global_updates,
                       "stage_updates": scheduler.updates,
                       "train_loss": loss_total / examples,
                       "learning_rate": optimizer.param_groups[0]["lr"]}
            if stage_name == "finetune":
                metrics.update(evaluate(model, dev_loader, device, world))
                if selection_improves(metrics["valid_bedroc"], best_bedroc):
                    best_bedroc = metrics["valid_bedroc"]
                    best_epoch = stage_epoch
                    patience_counter = 0
                    if rank == 0:
                        atomic_torch_save({"state_dict": model.state_dict(),
                                           "model_config": config.to_dict()},
                                          work / "finetune_best.pt")
                else:
                    patience_counter += 1
                metrics["best_bedroc"] = best_bedroc
                metrics["patience_counter"] = patience_counter
            completed[stage_name] = stage_epoch
            elapsed = (elapsed_before + time.monotonic() - started
                       if rank == 0 else 0.0)
            decision = (stop_decision(
                stage=stage_name, epoch=stage_epoch, max_epochs=max_epochs,
                another_stage=stage_name == "profsa" and args.stage == "both",
                elapsed=elapsed, max_seconds=args.max_seconds,
                patience_counter=patience_counter, patience=args.patience)
                if rank == 0 else None)
            decision, elapsed = broadcast_stop(decision, elapsed,
                                               device=device, world=world)
            if decision in ("time_budget", "patience", "max_epochs"):
                stop_reason = decision
            metrics["elapsed_seconds"] = elapsed
            metrics["stop_decision"] = decision
            if device.type == "cuda":
                metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            if rank == 0:
                print(compact_json(metrics), flush=True)
                with (work / "metrics.jsonl").open("a") as stream:
                    stream.write(compact_json(metrics) + "\n")
            rank_states = gather_rng(world)
            if rank == 0:
                atomic_torch_save({
                    "identity": identity, "stage": stage_name, "epoch": stage_epoch,
                    "completed_epochs": completed.copy(), "global_updates": global_updates,
                    "completed_updates": completed_updates.copy(),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "best_bedroc": best_bedroc, "best_epoch": best_epoch,
                    "patience_counter": patience_counter, "elapsed_seconds": elapsed,
                    "rng_by_rank": rank_states, "stop_reason": decision,
                }, work / "last.pt")
                if stage_name == "profsa" and stage_epoch == max_epochs:
                    atomic_torch_save({"stage": "profsa", "complete": True,
                                       "epoch": max_epochs, "model": model.state_dict(),
                                       "updates": completed_updates["profsa"],
                                       "samples": len(dataset),
                                       "identity": identity}, work / "profsa_complete.pt")
            if world > 1:
                dist.barrier()
            if decision in ("time_budget", "patience", "max_epochs"):
                break
        if world > 1:
            dist.barrier()
        del wrapped
        if stop_reason in ("time_budget", "patience"):
            break
    if world > 1:
        dist.barrier()
    if rank == 0:
        summary = dict(identity, completed_epochs=completed,
                       completed_updates=completed_updates,
                       global_updates=global_updates, stop_reason=stop_reason,
                       patience_counter=patience_counter,
                       elapsed_seconds=elapsed_before + time.monotonic() - started,
                       peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                  if device.type == "cuda" else None),
                       selected_stage="finetune" if best_epoch else None,
                       selected_epoch=best_epoch,
                       best_valid_bedroc=best_bedroc if best_epoch else None)
        if best_epoch is not None:
            summary["selected_checkpoint"] = "model.pt"
            summary["finetune_best_sha256"] = sha256(work / "finetune_best.pt")
            def build(stage: Path) -> None:
                for name in ("model.py", "unimol_encoder.py", "data.py",
                             "common.py", "predict.py", "validation.py",
                             "training.py", "infer.sh", "requirements.txt"):
                    shutil.copyfile(source_dir / name, stage / name)
                (stage / "infer.sh").chmod(0o755)
                shutil.copyfile(work / "finetune_best.pt", stage / "model.pt")
                (stage / "model_config.json").write_text(
                    json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
                (stage / "training_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n")
                (stage / "model_manifest.json").write_text(json.dumps({
                    "schema": "target-based-drug-screening-model/v2",
                    "method": "paper-dual-unimol",
                    "checkpoint": {"path": "model.pt", "sha256": sha256(stage / "model.pt")},
                    "model_config": "model_config.json", "parameter_count": count,
                    "training_summary": "training_summary.json",
                }, indent=2, sort_keys=True) + "\n")
            publish_directory(output, build)
        (work / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

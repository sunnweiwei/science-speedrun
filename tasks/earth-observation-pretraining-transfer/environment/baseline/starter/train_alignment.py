"""WorldCover-supervised scratch starter, with optional spectral supervision."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ("sentinel2_l2a", "sentinel1")
ARRAYS = (
    *MODALITIES,
    "sentinel2_l2a_valid",
    "sentinel1_valid",
    "worldcover_fractions",
    "worldcover_weight",
    "spectral_moments",
    "spectral_moments_valid",
)
REVISION = "0497dfbb6711ded4e6bf10cf089fc1e4d58c186b"
WORLDCOVER_PARENT_SCHEDULE = {
    "type": "cosine", "horizon": 6000, "warmup": 300, "final_ratio": 0.1
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scientific_parent_recipe(recipe):
    """Keep the Phase 1 science comparable while storing artifact identity separately."""
    result = copy.deepcopy(recipe)
    result.pop("init_seed", None)
    result.pop("data_seed", None)
    return result


def valid_stage_parent_step(step):
    """Accept saved learning progress within the unchanged WorldCover schedule."""
    return type(step) is int and 0 < step <= WORLDCOVER_PARENT_SCHEDULE["horizon"]


def load_stage_parent(path, expected_sha256):
    """Verify the immutable, locally trained WorldCover boundary and its archived sources."""
    path = Path(path).resolve()
    if not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError("Stage parent checkpoint hash mismatch")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if (
        sidecar.get("sha256") != expected_sha256
        or not sidecar.get("recipe_complete")
    ):
        raise ValueError("Stage parent completion receipt mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    recipe = state.get("recipe", {})
    parent_step = state.get("global_step")
    if not valid_stage_parent_step(parent_step):
        raise ValueError("Stage parent update must be an integer in the WorldCover range 1..6000")
    if (
        type(sidecar.get("global_step")) is not int
        or sidecar["global_step"] != parent_step
    ):
        raise ValueError("Stage parent completion receipt step mismatch")
    if (
        state.get("kind") != "olmoearth_alignment"
        or state.get("format_version") != 1
        or recipe.get("origin") != "scratch"
        or recipe.get("parent_checkpoint") is not None
        or recipe.get("model_variant") != "large"
        or recipe.get("encoder_config") != state.get("encoder_config")
        or recipe.get("loss", {}).get("worldcover_weight") != 1.0
        or recipe.get("loss", {}).get("moments_weight") != 0.0
        or recipe.get("loss", {}).get("heads") != "linear_11_24_4"
        or recipe.get("schedule") != WORLDCOVER_PARENT_SCHEDULE
    ):
        raise ValueError("Stage parent must be a completed scratch Large WorldCover checkpoint")
    archives = sorted((path.parent.parent / "launches").glob("*/source-trainer.py"))
    archive = next(
        (p for p in archives if sha256(p) == recipe["source_hashes"]["trainer"]), None
    )
    if archive is None:
        raise ValueError("Stage parent archived trainer does not match its recipe")
    lineage = {
        "parent_checkpoint": str(path),
        "parent_sha256": expected_sha256,
        "parent_global_step": parent_step,
        "parent_init_seed": recipe["init_seed"],
        "parent_data_seed": recipe["data_seed"],
        "parent_trainer_snapshot": str(archive),
        "parent_trainer_sha256": recipe["source_hashes"]["trainer"],
        "root_origin": "own_from_scratch_training",
    }
    return state, lineage


def validate_parent_compatibility(parent, recipe, temporal_order="source"):
    """Allow the declared stage changes, retaining model/data/evaluation identity."""
    if temporal_order not in ("source", "newest-first"):
        raise ValueError("Unsupported declared stage temporal order")
    old = parent["recipe"]
    for key in (
        "upstream", "upstream_diff_sha256", "model_variant", "encoder_config",
        "init_seed", "data_seed", "cache_manifest", "cache_manifest_sha256",
        "cache_rows", "world_size", "global_batch", "microbatch", "accumulation",
        "precision", "input", "readout", "sampler",
    ):
        candidate = recipe.get(key)
        if key == "input" and temporal_order == "newest-first":
            candidate = copy.deepcopy(candidate)
            if candidate.pop("temporal_order", None) != "newest-first":
                raise ValueError("Missing declared newest-first input transition")
        if old.get(key) != candidate:
            raise ValueError(f"Stage parent compatibility mismatch: {key}")
    for name, digest in old["source_hashes"].items():
        if name != "trainer" and recipe["source_hashes"].get(name) != digest:
            raise ValueError(f"Stage parent source mismatch: {name}")


def initialize_moisture_head(heads, width, outputs, seed, device):
    """Initialize only the new head, without consuming the encoder's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = torch.nn.Linear(width, outputs)
    heads["moisture"] = head.to(device)


def moisture_denominator(batch):
    return float((batch["moisture_valid"] * batch["moisture_weight"][..., None]).sum(dtype=np.float64))


def moisture_loss(features, heads, batch, device, mean, std, denominator):
    target = torch.as_tensor(batch["moisture_targets"], device=device)
    valid = torch.as_tensor(batch["moisture_valid"], device=device, dtype=torch.bool)
    weight = torch.as_tensor(batch["moisture_weight"], device=device)[..., None]
    error = (heads["moisture"](features.float()) - (target - mean) / std).square()
    return (torch.where(valid, error, 0.0) * weight).sum() / (
        denominator if denominator > 0 else 1.0
    )


def order_training_batch(batch, temporal_order):
    """Reverse input observations once; cached targets already have their final order."""
    if temporal_order == "source":
        return batch
    if temporal_order != "newest-first":
        raise ValueError("Unsupported training temporal order")
    result = dict(batch)
    for name in MODALITIES:
        result[name] = np.ascontiguousarray(np.flip(batch[name], axis=3))
        result[name + "_valid"] = np.ascontiguousarray(
            np.flip(batch[name + "_valid"], axis=1)
        )
    return result


def build_encoder(upstream, seed, variant="base"):
    """Match export.py's complete official builder and initialization order."""
    if variant not in ("base", "large"):
        raise ValueError(f"Unsupported alignment encoder variant: {variant}")
    script_dir = upstream / "scripts/official/v1"
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(script_dir))
    builder = importlib.import_module(variant)
    script = importlib.import_module("script")
    common = script.build_common_components(
        str(script_dir / f"{variant}.py"), "dry_run", "export", "local", []
    )
    config = builder.build_model_config(common)
    torch.manual_seed(seed)
    full_model = config.build()
    encoder = full_model.encoder
    del full_model
    return encoder, config.encoder_config.as_config_dict()


class AlignmentCache:
    """Read-only mmap arrays with an exact sealed training manifest."""

    def __init__(self, manifest_path, verify_hashes=True):
        self.path = Path(manifest_path).resolve()
        self.manifest = m = json.loads(self.path.read_text())
        self.manifest_hash = sha256(self.path)
        if m["patch_size"] != 4 or m["sequence_length"] != 12:
            raise ValueError("Require patch4 and twelve months")
        if m["worldcover_classes"] != [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
            raise ValueError("Unexpected WorldCover categories")
        self.arrays = {}
        for name in ARRAYS:
            spec = m["arrays"][name]
            path = (self.path.parent / spec["path"]).resolve()
            if not path.is_relative_to(self.path.parent):
                raise ValueError("Arrays must remain in the cache directory")
            if verify_hashes and sha256(path) != spec["sha256"]:
                raise ValueError(f"Cache hash mismatch: {name}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise ValueError(f"Cache array declaration mismatch: {name}")
            self.arrays[name] = value
        self.size = n = len(self.arrays[MODALITIES[0]])
        crop = m["crop_size"]
        if crop < 4 or crop % 4:
            raise ValueError("Crop must be a positive multiple of patch4")
        grid = crop // 4
        shapes = {
            "sentinel2_l2a": (n, crop, crop, 12, 12),
            "sentinel1": (n, crop, crop, 12, 2),
            "sentinel2_l2a_valid": (n, 12, 3),
            "sentinel1_valid": (n, 12, 1),
            "worldcover_fractions": (n, grid, grid, 11),
            "worldcover_weight": (n, grid, grid),
            "spectral_moments": (n, grid, grid, 28),
            "spectral_moments_valid": (n, grid, grid, 28),
        }
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"Invalid shape for {name}: expected {shape}")
        observed = np.zeros(n, dtype=bool)
        for name in MODALITIES:
            observed |= self.arrays[name + "_valid"].reshape(n, -1).any(1)
        if not observed.all():
            raise ValueError("Cache includes rows with no source observations")
        self.moment_mean = np.asarray(m["moment_mean"], dtype=np.float32)
        self.moment_std = np.asarray(m["moment_std"], dtype=np.float32)
        for value in (self.moment_mean, self.moment_std):
            if value.shape != (28,) or not np.isfinite(value).all():
                raise ValueError("Expected 28 finite training-only moment statistics")
        if (self.moment_std < 0).any():
            raise ValueError("Negative moment standard deviation")

    def batch(self, indices, augment):
        batch = {
            name: np.asarray(value[indices]) for name, value in self.arrays.items()
        }
        if augment:
            rotations, flip = np.random.randint(4), bool(np.random.randint(2))
            for name, value in batch.items():
                if name.endswith("_valid") and name not in (
                    "spectral_moments_valid", "moisture_valid"
                ):
                    continue
                value = np.rot90(value, rotations, axes=(1, 2))
                if flip:
                    value = np.flip(value, axis=1)
                batch[name] = np.ascontiguousarray(value)
        return batch


class BatchSampler:
    """Worker-free shuffled sampler with exact checkpointed progress."""

    def __init__(self, size, batch_size, seed):
        if size < batch_size:
            raise ValueError("Cache must contain a complete global batch")
        self.size, self.batch_size = size, batch_size
        self.generator = np.random.default_rng(seed)
        self.epoch, self.next_offset = 0, 0
        self.indices = np.empty(0, dtype=np.int64)

    def next(self):
        if self.next_offset + self.batch_size > len(self.indices):
            self.indices = self.generator.permutation(self.size)
            self.next_offset = 0
            self.epoch += 1
        result = self.indices[self.next_offset : self.next_offset + self.batch_size]
        self.next_offset += self.batch_size
        return result

    def state_dict(self):
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "indices": self.indices,
            "next_offset": self.next_offset,
            "generator_state": self.generator.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Resume sampler/cache mismatch")
        self.epoch, self.indices, self.next_offset = (
            state["epoch"],
            state["indices"],
            state["next_offset"],
        )
        self.generator.bit_generator.state = state["generator_state"]


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def lr_factor(step, warmup, horizon, final_ratio):
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / (horizon - warmup)
    return final_ratio + (1 - final_ratio) * (1 + math.cos(math.pi * progress)) / 2


def encoder_spatial_features(encoder, batch, device):
    """Canonical native spatial mean and per-sample modality correction."""
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
    from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

    kwargs, validity, fast_pass = {}, [], True
    size = len(batch[MODALITIES[0]])
    for name in MODALITIES:
        pixels = torch.from_numpy(batch[name]).to(device=device, dtype=torch.float32)
        valid = torch.from_numpy(batch[name + "_valid"]).to(
            device=device, dtype=torch.bool
        )
        validity.append(valid.reshape(size, -1).any(1))
        fast_pass = fast_pass and bool(valid.all())
        mask = torch.zeros(
            (*pixels.shape[:-1], valid.shape[-1]), dtype=torch.int32, device=device
        )
        mask.masked_fill_(~valid[:, None, None, :, :], 3)
        kwargs[name], kwargs[name + "_mask"] = pixels, mask
    timestamps = torch.zeros((size, 12, 3), dtype=torch.int32, device=device)
    timestamps[:, :, 0], timestamps[:, :, 2] = 1, 2024
    timestamps[:, :, 1] = torch.arange(12, device=device)
    kwargs["timestamps"] = timestamps
    result = encoder(MaskedOlmoEarthSample(**kwargs), patch_size=4, fast_pass=fast_pass)
    features = pool_unmasked_tokens(
        result["tokens_and_masks"], PoolingType.MEAN, spatial_pooling=True
    )
    present = torch.stack(validity, dim=1)
    if not bool(present.any(1).all()):
        raise ValueError("Training row has no observations")
    scale = present.any(0).sum() / present.sum(1)
    return features * scale.to(features.dtype)[:, None, None, None]


def loss_denominators(batch):
    return {
        "worldcover": float(batch["worldcover_weight"].sum()),
        "s2_moments": int(batch["spectral_moments_valid"][..., :24].sum()),
        "s1_moments": int(batch["spectral_moments_valid"][..., 24:].sum()),
    }


def alignment_losses(features, heads, batch, device, mean, std, den):
    """Global denominators make gradient accumulation mathematically consistent."""
    fractions = torch.as_tensor(batch["worldcover_fractions"], device=device)
    coverage = torch.as_tensor(batch["worldcover_weight"], device=device)
    moments = torch.as_tensor(batch["spectral_moments"], device=device)
    valid = torch.as_tensor(
        batch["spectral_moments_valid"], device=device, dtype=torch.bool
    )
    features = features.float()
    logits = heads["worldcover"](features)
    wc = (-(fractions * F.log_softmax(logits, dim=-1)).sum(-1) * coverage).sum() / max(
        den["worldcover"], 1.0
    )
    predictions = torch.cat(
        (heads["s2_moments"](features), heads["s1_moments"](features)), dim=-1
    )
    error = torch.where(valid, (predictions - (moments - mean) / std).square(), 0.0)
    s2 = error[..., :24].sum() / max(den["s2_moments"], 1)
    s1 = error[..., 24:].sum() / max(den["s1_moments"], 1)
    count = int(den["s2_moments"] > 0) + int(den["s1_moments"] > 0)
    return {
        "worldcover": wc,
        "spectral_moments": (s2 + s1) / max(count, 1),
        "s2_moments": s2,
        "s1_moments": s1,
    }


def gradient_norm(parameters):
    values = [
        p.grad.detach().float().square().sum() for p in parameters if p.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def cpu_state(module):
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def save_checkpoint(output, step, encoder, heads, optimizer, sampler, recipe, metrics,
                    lineage=None):
    path = output / "checkpoints" / f"step{step:06d}.pt"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    state = {
        "format_version": 1,
        "kind": "olmoearth_alignment",
        "global_step": step,
        "encoder": cpu_state(encoder),
        "encoder_config": recipe["encoder_config"],
        "heads": cpu_state(heads),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"global_step": step, **recipe["schedule"]},
        "rng": rng_state(),
        "sampler": sampler.state_dict(),
        "recipe": recipe,
        "metrics_last": metrics,
    }
    if lineage is not None:
        state["lineage"] = copy.deepcopy(lineage)
        state["total_encoder_updates"] = lineage["parent_global_step"] + step
    start = time.perf_counter()
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    digest = sha256(path)
    atomic_json(
        path.with_suffix(".json"),
        {
            "checkpoint": str(path),
            "sha256": digest,
            "global_step": step,
            "completed_utc": utc_now(),
            "bytes": path.stat().st_size,
            "save_and_hash_seconds": time.perf_counter() - start,
            "recipe_complete": True,
            "metrics_last": metrics,
        },
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_complete",
                "step": step,
                "path": str(path),
                "sha256": digest,
            }
        ),
        flush=True,
    )
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "cache-manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variant", choices=("base", "large"), default="base")
    for name, default in (
        ("init-seed", 12536),
        ("data-seed", 3622),
        ("steps", 6000),
        ("warmup", 300),
        ("global-batch", 64),
        ("microbatch", 16),
        ("save-every", 1000),
        ("log-every", 10),
    ):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (
        ("lr", 3e-5),
        ("head-lr", 1e-3),
        ("weight-decay", 0.002),
        ("max-grad-norm", 1.0),
        ("final-lr-ratio", 0.1),
        ("worldcover-weight", 1.0),
        ("moments-weight", 0.0),
        ("moisture-weight", 0.0),
        ("moment-std-floor", 0.05),
    ):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument(
        "--stop-at",
        type=int,
        default=1000,
        help="Selected starter checkpoint; --steps remains the cosine horizon",
    )
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-from", type=Path)
    p.add_argument("--parent-sha256")
    p.add_argument("--moisture-manifest", type=Path)
    p.add_argument("--temporal-order", choices=("source", "newest-first"), default="source")
    p.add_argument("--head-init-seed", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--skip-cache-hash-verification",
        action="store_true",
        help="Only after separately recording verification of this exact cache",
    )
    p.add_argument("--check-cache-only", action="store_true")
    a = p.parse_args()
    if not 0 < a.stop_at <= a.steps or not 0 < a.warmup < a.steps:
        p.error("Require 0 < stop-at <= steps and 0 < warmup < steps")
    if a.global_batch <= 0 or a.microbatch <= 0 or a.global_batch % a.microbatch:
        p.error("Global batch must be a positive multiple of microbatch")
    if (
        min(
            a.save_every,
            a.log_every,
            a.moment_std_floor,
            a.lr,
            a.head_lr,
            a.max_grad_norm,
        )
        <= 0
    ):
        p.error("Intervals, learning rates, clip and std floor must be positive")
    if a.weight_decay < 0 or not 0 <= a.final_lr_ratio <= 1:
        p.error("Invalid regularization or final LR ratio")
    if (
        min(a.worldcover_weight, a.moments_weight, a.moisture_weight) < 0
        or a.worldcover_weight + a.moments_weight + a.moisture_weight <= 0
    ):
        p.error("At least one positive loss weight required")
    if a.initialize_from and not a.parent_sha256:
        p.error("--initialize-from requires --parent-sha256")
    if a.parent_sha256 and not a.initialize_from:
        p.error("--parent-sha256 is only used with --initialize-from")
    if bool(a.moisture_manifest) != bool(a.moisture_weight > 0):
        p.error("A positive moisture weight and --moisture-manifest are required together")
    if a.initialize_from and not a.moisture_manifest:
        p.error("Stage initialization requires --moisture-manifest")
    if a.moisture_manifest and not (a.initialize_from or a.resume):
        p.error("Moisture training requires a declared stage parent or stage resume")
    if a.head_init_seed is not None and not a.moisture_manifest:
        p.error("--head-init-seed is only used with moisture training")
    if a.temporal_order != "source" and not a.moisture_manifest:
        p.error("Temporal ordering changes require declared moisture-stage training")
    return a


def main():
    a = parse_args()
    started, started_utc = time.perf_counter(), utc_now()
    upstream = a.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Unexpected upstream revision")
    cache = AlignmentCache(
        a.cache_manifest, verify_hashes=not a.skip_cache_hash_verification
    )
    moisture = None
    if a.moisture_manifest:
        from prepare_moisture import MoistureCache

        moisture = MoistureCache(a.moisture_manifest, cache)
        cache.arrays.update(moisture.arrays)
    if a.check_cache_only:
        print(
            json.dumps(
                {
                    "rows": cache.size,
                    "cache_manifest_sha256": cache.manifest_hash,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        )
        return
    device = torch.device(a.device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Alignment trainer supports world size one")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        if (
            os.environ.get("EARTH_SLURM_PARTITION")
            and os.environ.get("SLURM_JOB_PARTITION")
            != os.environ["EARTH_SLURM_PARTITION"]
        ):
            raise RuntimeError("GPU training requires the configured Slurm partition")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output = a.output.resolve()
    if not a.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Scratch training requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(a.data_seed)
    np.random.seed(a.data_seed)
    encoder, encoder_config = build_encoder(upstream, a.init_seed, a.variant)
    encoder.to(device).train()
    heads = torch.nn.ModuleDict(
        {
            name: torch.nn.Linear(encoder.embedding_size, width)
            for name, width in (
                ("worldcover", 11),
                ("s2_moments", 24),
                ("s1_moments", 4),
            )
        }
    ).to(device)
    if moisture is not None:
        initialize_moisture_head(
            heads, encoder.embedding_size, moisture.target_width,
            a.init_seed if a.head_init_seed is None else a.head_init_seed, device,
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(encoder.parameters()),
                "lr": a.lr,
                "initial_lr": a.lr,
                "name": "encoder",
            },
            {
                "params": list(heads.parameters()),
                "lr": a.head_lr,
                "initial_lr": a.head_lr,
                "name": "heads",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=a.weight_decay,
    )
    sampler = BatchSampler(cache.size, a.global_batch, a.data_seed)
    diff = subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--binary"]
    )
    source_files = {
        "trainer": Path(__file__).resolve(),
        "infer": Path(__file__).with_name("infer.py").resolve(),
        f"{a.variant}_builder": upstream / f"scripts/official/v1/{a.variant}.py",
        "official_script": upstream / "scripts/official/v1/script.py",
        "encoder": upstream / "olmoearth_pretrain/nn/flexi_vit.py",
        "pooling": upstream / "olmoearth_pretrain/nn/pooling.py",
        "normalization": upstream / "olmoearth_pretrain/data/normalize.py",
    }
    if moisture is not None:
        source_files["moisture_targets"] = Path(__file__).with_name("prepare_moisture.py").resolve()
    recipe = {
        "recipe_complete": True,
        "kind": "alignment_worldcover_spectral_moments_v1",
        "origin": "scratch",
        "parent_checkpoint": None,
        "upstream": revision,
        "upstream_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "model_variant": a.variant,
        "encoder_config": encoder_config,
        "init_seed": a.init_seed,
        "data_seed": a.data_seed,
        "cache_manifest": str(cache.path),
        "cache_manifest_sha256": cache.manifest_hash,
        "cache_rows": cache.size,
        "source_hashes": {name: sha256(path) for name, path in source_files.items()},
        "world_size": 1,
        "global_batch": a.global_batch,
        "microbatch": a.microbatch,
        "accumulation": a.global_batch // a.microbatch,
        "precision": (
            "float32_parameters_bfloat16_autocast"
            if device.type == "cuda"
            else "float32"
        ),
        "optimizer": {
            "type": "AdamW",
            "encoder_lr": a.lr,
            "head_lr": a.head_lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": a.weight_decay,
            "max_grad_norm": a.max_grad_norm,
        },
        "schedule": {
            "type": "cosine",
            "horizon": a.steps,
            "warmup": a.warmup,
            "final_ratio": a.final_lr_ratio,
        },
        "loss": {
            "worldcover": "coverage_weighted_soft_cross_entropy",
            "worldcover_weight": a.worldcover_weight,
            "spectral_moments": "equal_modality_standardized_mse",
            "moments_weight": a.moments_weight,
            "moment_std_floor": a.moment_std_floor,
            "moment_mean": cache.moment_mean.tolist(),
            "moment_std": cache.moment_std.tolist(),
            "heads": "linear_11_24_4",
        },
        "input": {
            "modalities": list(MODALITIES),
            "patch_size": 4,
            "masking": "present_observations_unmasked",
            "normalization": "cache_COMPUTED_S1_already_decibels",
            "timestamps": "day1_month0to11_year2024",
            "crop_size": cache.manifest["crop_size"],
            "temporal_length": 12,
            "augment": "none" if a.no_augment else "joint_batch_flip_and_rot90",
        },
        "readout": "canonical_native_spatial_mean_with_per_sample_modality_correction",
        "sampler": {
            "shuffle": True,
            "drop_last": True,
            "workers": 0,
            "exact_rng_continuation": True,
        },
    }
    if a.temporal_order != "source":
        recipe["input"]["temporal_order"] = a.temporal_order
    lineage, stage_state = None, None
    if moisture is not None:
        if a.initialize_from:
            stage_state, lineage = load_stage_parent(a.initialize_from, a.parent_sha256)
            validate_parent_compatibility(stage_state, recipe, temporal_order=a.temporal_order)
            parent_science = {
                "global_step": stage_state["global_step"],
                "recipe": scientific_parent_recipe(stage_state["recipe"]),
            }
        else:
            stage_state = torch.load(a.resume, map_location="cpu", weights_only=False)
            lineage = copy.deepcopy(stage_state.get("lineage"))
            parent_science = copy.deepcopy(stage_state.get("recipe", {}).get("parent_checkpoint"))
            if (
                not lineage or not parent_science
                or lineage.get("root_origin") != "own_from_scratch_training"
                or not valid_stage_parent_step(lineage.get("parent_global_step"))
                or lineage.get("parent_init_seed") != a.init_seed
                or lineage.get("parent_data_seed") != a.data_seed
                or not valid_stage_parent_step(parent_science.get("global_step"))
                or parent_science["global_step"] != lineage["parent_global_step"]
                or parent_science.get("recipe", {}).get("origin") != "scratch"
                or parent_science.get("recipe", {}).get("schedule") != WORLDCOVER_PARENT_SCHEDULE
            ):
                raise ValueError("Invalid staged resume lineage")
        recipe.update(
            kind="alignment_worldcover_moisture_stage_v1",
            origin="project_scratch_branch",
            parent_checkpoint=parent_science,
            head_init_seed=a.init_seed if a.head_init_seed is None else a.head_init_seed,
            moisture_cache={
                "manifest": str(moisture.path),
                "manifest_sha256": moisture.manifest_hash,
                "declaration": moisture.manifest,
            },
            transition={
                "encoder": "strict_parent_state",
                "existing_heads": "strict_parent_state",
                "new_head": "isolated_cpu_linear_initialization",
                "optimizer": "fresh_AdamW_all_parameters",
                "schedule": "new_phase_local_schedule",
                "sampler": "restore_parent",
                "rng": "restore_parent_after_new_head_initialization",
                "step_count": "phase_local_with_separate_total_encoder_updates",
            },
        )
        recipe["loss"].update(
            moisture="weighted_masked_standardized_mse",
            moisture_weight=a.moisture_weight,
            moisture_std_floor=moisture.manifest["recipe"]["target_std_floor"],
            moisture_mean=moisture.mean.tolist(),
            moisture_std=moisture.std.tolist(),
            heads=f"linear_11_24_4_plus_moisture_{moisture.target_width}",
        )
        if a.temporal_order != "source":
            recipe["loss"]["moisture_objective"] = "moments"
            recipe["transition"]["temporal_order"] = "reverse_source_inputs_oldest_to_newest_into_newest_first"
    step, metrics_last = 0, {}
    if a.resume:
        state = stage_state if stage_state is not None else torch.load(
            a.resume, map_location="cpu", weights_only=False
        )
        if state.get("kind") != "olmoearth_alignment" or state.get("recipe") != recipe:
            raise ValueError("Checkpoint kind or complete continuation recipe mismatch")
        step = state["global_step"]
        if not step < a.stop_at:
            raise ValueError("Resume stop must exceed the saved step")
        encoder.load_state_dict(state["encoder"], strict=True)
        heads.load_state_dict(state["heads"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
        metrics_last = state["metrics_last"]
        del state
    else:
        if stage_state is not None:
            encoder.load_state_dict(stage_state["encoder"], strict=True)
            for name in ("worldcover", "s2_moments", "s1_moments"):
                prefix = name + "."
                heads[name].load_state_dict(
                    {key[len(prefix):]: value for key, value in stage_state["heads"].items()
                     if key.startswith(prefix)}, strict=True,
                )
            sampler.load_state_dict(stage_state["sampler"])
            restore_rng(stage_state["rng"])
        atomic_json(output / "recipe.json", recipe)
    del stage_state
    launch = (
        output / "launches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    launch.mkdir(parents=True)
    for name, path in source_files.items():
        shutil.copy2(path, launch / f"source-{name}.py")
    (launch / "upstream.diff").write_bytes(diff)
    shutil.copy2(cache.path, launch / "cache-manifest.json")
    if moisture is not None:
        shutil.copy2(moisture.path, launch / "moisture-manifest.json")
        atomic_json(launch / "lineage.json", lineage)
    atomic_json(
        launch / "launch.json",
        {
            "pid": os.getpid(),
            "command": sys.argv,
            "start_utc": started_utc,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "resume": str(a.resume.resolve()) if a.resume else None,
            "resume_sha256": sha256(a.resume) if a.resume else None,
            "start_step": step,
            "stop_at": a.stop_at,
            "cache_hashes_verified_here": not a.skip_cache_hash_verification,
            "recipe": recipe,
        },
    )
    if not a.resume:
        save_checkpoint(output, 0, encoder, heads, optimizer, sampler, recipe, {},
                        **({"lineage": lineage} if lineage is not None else {}))
    mean = torch.tensor(cache.moment_mean, device=device)
    std = torch.tensor(np.maximum(cache.moment_std, a.moment_std_floor), device=device)
    if moisture is not None:
        moisture_mean = torch.tensor(moisture.mean, device=device)
        moisture_std = torch.tensor(moisture.std, device=device)
    parameters = list(encoder.parameters()) + list(heads.parameters())
    log = (launch / "metrics.jsonl").open("a", buffering=1)
    start_step, training_start, status = step, time.perf_counter(), "running"
    try:
        while step < a.stop_at:
            step_started = time.perf_counter()
            next_step = step + 1
            batch = cache.batch(sampler.next(), augment=not a.no_augment)
            batch = order_training_batch(batch, a.temporal_order)
            den = loss_denominators(batch)
            if moisture is not None:
                den["moisture"] = moisture_denominator(batch)
            if den["worldcover"] == 0 and den["s2_moments"] + den["s1_moments"] == 0:
                raise ValueError("Batch has no auxiliary supervision")
            factor = lr_factor(next_step, a.warmup, a.steps, a.final_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            metrics = {
                name: 0.0
                for name in (
                    "worldcover",
                    "spectral_moments",
                    "s2_moments",
                    "s1_moments",
                    "total_loss",
                )
            }
            if moisture is not None:
                metrics["moisture"] = 0.0
            feature_std = 0.0
            for offset in range(0, a.global_batch, a.microbatch):
                micro = {
                    name: value[offset : offset + a.microbatch]
                    for name, value in batch.items()
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    features = encoder_spatial_features(encoder, micro, device)
                losses = alignment_losses(
                    features, heads, micro, device, mean, std, den
                )
                loss = (
                    a.worldcover_weight * losses["worldcover"]
                    + a.moments_weight * losses["spectral_moments"]
                )
                if moisture is not None:
                    losses["moisture"] = moisture_loss(
                        features, heads, micro, device, moisture_mean, moisture_std,
                        den["moisture"],
                    )
                    loss = loss + a.moisture_weight * losses["moisture"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Nonfinite alignment objective at step {next_step}"
                    )
                loss.backward()
                for name, value in losses.items():
                    metrics[name] += float(value.detach())
                metrics["total_loss"] += float(loss.detach())
                feature_std += (
                    float(
                        features.detach()
                        .float()
                        .flatten(0, 2)
                        .std(0, unbiased=False)
                        .mean()
                    )
                    / recipe["accumulation"]
                )
                del features, losses, loss
            log_step = (
                next_step == 1 or next_step % a.log_every == 0 or next_step == a.stop_at
            )
            if log_step:
                metrics["encoder_grad_norm"] = gradient_norm(encoder.parameters())
                metrics["heads_grad_norm"] = gradient_norm(heads.parameters())
                if moisture is not None:
                    metrics["moisture_head_grad_norm"] = gradient_norm(heads["moisture"].parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, a.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            step = next_step
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics.update(
                {
                    "step": step,
                    "epoch": sampler.epoch,
                    "utc": utc_now(),
                    "encoder_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                    "grad_norm": float(grad_norm),
                    "pooled_feature_std_mean": feature_std,
                    "step_seconds": time.perf_counter() - step_started,
                }
            )
            metrics.update(
                {name + "_denominator": value for name, value in den.items()}
            )
            if lineage is not None:
                metrics["total_encoder_updates"] = lineage["parent_global_step"] + step
            metrics_last = metrics
            if log_step:
                log.write(json.dumps(metrics) + "\n")
                print(json.dumps({"event": "training", **metrics}), flush=True)
            if step % a.save_every == 0 or step == a.stop_at:
                save_checkpoint(
                    output, step, encoder, heads, optimizer, sampler, recipe, metrics,
                    **({"lineage": lineage} if lineage is not None else {}),
                )
        status = "completed"
    finally:
        log.close()
        elapsed = time.perf_counter() - started
        atomic_json(
            launch / "completion.json",
            {
                "status": status,
                "pid": os.getpid(),
                "start_utc": started_utc,
                "end_utc": utc_now(),
                "global_step": step,
                "updates_this_launch": step - start_step,
                "wall_seconds": elapsed,
                "training_loop_seconds": time.perf_counter() - training_start,
                "assigned_gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                "peak_cuda_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                ),
                "metrics_last": metrics_last,
            },
        )


if __name__ == "__main__":
    main()

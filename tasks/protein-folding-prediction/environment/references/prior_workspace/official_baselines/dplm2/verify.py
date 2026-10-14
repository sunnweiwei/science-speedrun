#!/usr/bin/env python3
"""Verify source provenance and topology of the official-code DPLM-2 scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_baselines.dplm2.config import (
    EXPECTED_FOLDING_CORE_RANGE,
    LOCKED_RECIPE,
    LOCKED_SEQUENCE_RECIPES,
    MODEL_NAME,
    SCALED_ARCHITECTURE,
    STRUCT_TOKENIZER_MODEL,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("/opt/dplm"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    head = (repository / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = repository / ".git" / ref
    if loose.exists():
        return loose.read_text().strip()
    for line in (repository / ".git" / "packed-refs").read_text().splitlines():
        if line and not line.startswith(("#", "^")):
            revision, name = line.split(" ", 1)
            if name == ref:
                return revision
    raise RuntimeError(f"cannot resolve revision under {repository}")


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def verify_recipe(upstream: Path) -> str:
    """Check locked fields against the actual YAML at the pinned revision."""
    from omegaconf import OmegaConf

    path = upstream / "configs/experiment/dplm2/dplm2_150m.yaml"
    config = OmegaConf.load(path)
    actual = {
        "num_diffusion_timesteps": config.model.num_diffusion_timesteps,
        "single_modality_ratio": config.model.single_modality_ratio,
        "folding_loss_ratio": config.model.folding_loss_ratio,
        "inverse_folding_loss_ratio": config.model.inverse_folding_loss_ratio,
        "joint_loss_ratio": config.model.joint_loss_ratio,
        "independent_loss_ratio": config.model.independent_loss_ratio,
        "lora_rank": config.model.lora.lora_rank,
        "lora_dropout": config.model.lora.lora_dropout,
        "lora_alpha": 32,  # hard-coded by upstream get_net_dplm2
        "lora_target_module": config.model.lora.lora_target_module,
        "modules_to_save": config.model.lora.modules_to_save,
        "criterion": config.task.criterion._target_,
        "label_smoothing": config.task.criterion.label_smoothing,
        "cal_constant_loss": config.task.learning.cal_constant_loss,
        "learning_rate": config.train.lr,
        "adam_betas": list(config.task.optimizer.betas),
        "weight_decay": config.task.optimizer.weight_decay,
        "scheduler": config.task.lr_scheduler.type,
        "warmup_steps": config.task.lr_scheduler.warmup_steps,
        "total_steps": config.trainer.max_steps,
        "end_learning_rate": config.task.lr_scheduler.lr_end,
        "warmup_initial_learning_rate": config.task.lr_scheduler.warmup_init_lr,
        "gradient_clip": config.trainer.gradient_clip_val,
        "seed": config.train.seed,
    }
    if actual != LOCKED_RECIPE:
        differences = {
            key: {"expected": LOCKED_RECIPE.get(key), "actual": actual.get(key)}
            for key in sorted(set(actual) | set(LOCKED_RECIPE))
            if actual.get(key) != LOCKED_RECIPE.get(key)
        }
        raise AssertionError(f"upstream recipe mismatch: {differences}")
    return str(path)


def main() -> None:
    args = arguments()
    from transformers import AutoConfig
    from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2

    config = AutoConfig.from_pretrained(args.config, local_files_only=True)
    for key, expected in SCALED_ARCHITECTURE.items():
        actual = getattr(config, key)
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected}, got {actual}")
    if config.vocab_size != 8229 or config.dplm_type != "dplm2_esm":
        raise AssertionError("not a DPLM-2 8192-structure-token config")

    model = EsmForDPLM2(config, dropout=0.1, vocab_size=config.vocab_size)
    parameters = count(model)
    low, high = EXPECTED_FOLDING_CORE_RANGE
    if not low <= parameters <= high:
        raise AssertionError(
            f"folding-core parameter count {parameters:,} outside [{low:,}, {high:,}]"
        )
    revision = git_revision(args.upstream)
    if revision != UPSTREAM_REVISION:
        raise AssertionError(f"source revision {revision} != {UPSTREAM_REVISION}")
    recipe_source = verify_recipe(args.upstream)
    origins = sorted(
        {
            type(module).__module__
            for module in model.modules()
            if any(p.requires_grad for p in module.parameters(recurse=False))
        }
    )
    foreign = [
        origin
        for origin in origins
        if not (
            origin.startswith("byprot.models.dplm2.")
            or origin.startswith("transformers.models.esm.")
            or origin.startswith("torch.nn.")
        )
    ]
    if foreign:
        raise AssertionError(f"unexpected trainable module origins: {foreign}")
    report = {
        "schema_version": "official-folding-baseline-audit-v1",
        "model_name": MODEL_NAME,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": revision,
        "top_level_class": f"{type(model).__module__}.{type(model).__name__}",
        "folding_core_parameters": parameters,
        "external_frozen_structure_tokenizer": STRUCT_TOKENIZER_MODEL,
        "scaled_architecture": SCALED_ARCHITECTURE,
        "locked_training_recipe": LOCKED_RECIPE,
        "locked_sequence_pretraining_recipes": LOCKED_SEQUENCE_RECIPES,
        "recipe_source": recipe_source,
        "trainable_module_origins": origins,
        "foreign_trainable_module_origins": foreign,
        "capability": {
            "input": "single protein amino-acid sequence",
            "output": "protein structure tokens decoded to protein coordinates",
            "foldbench_all_nine_tracks": False,
            "reason": "the public DPLM-2 folding path is protein-only, not a general biomolecular all-atom predictor",
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

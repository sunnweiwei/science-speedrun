"""Export a scratch alignment checkpoint through the unchanged encoder interface."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from train_alignment import (
    load_stage_parent,
    scientific_parent_recipe,
    validate_parent_compatibility,
)

PINNED_REVISION = "0497dfbb6711ded4e6bf10cf089fc1e4d58c186b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_recipe(recipe, checkpoint_step):
    """Keep scientific settings invariant across independently seeded replicas.

    Seeds remain in package provenance and the saved checkpoint declaration.
    No other fields are dropped: architecture, data, losses, optimization,
    schedule and implementation hashes all participate in the recipe lock.
    The selected checkpoint update also participates, as in native exports.
    """
    result = copy.deepcopy(recipe)
    result.pop("init_seed", None)
    result.pop("data_seed", None)
    if result.get("origin") == "project_scratch_branch":
        result.pop("head_init_seed", None)
    result["checkpoint_step"] = f"step{checkpoint_step}"
    result["recipe_complete"] = True
    return result


def validate_checkpoint_lineage(saved):
    """Accept a verified local scratch ancestor; never reinterpret external weights."""
    recipe = saved["recipe"]
    if recipe.get("origin") == "scratch" and recipe.get("parent_checkpoint") is None:
        if saved.get("lineage") is not None:
            raise ValueError("Scratch checkpoint unexpectedly declares parent lineage")
        return None
    if recipe.get("origin") != "project_scratch_branch":
        raise ValueError("Expected project scratch origin or a verified local stage")
    if (
        set(recipe.get("schedule", {})) != {"type", "horizon", "warmup", "final_ratio"}
        or recipe["schedule"].get("type") != "cosine"
        or recipe.get("loss", {}).get("moisture") != "weighted_masked_standardized_mse"
        or recipe.get("loss", {}).get("heads") != "linear_11_24_4_plus_moisture_2"
    ):
        raise ValueError("Expected a cosine-scheduled moisture-moments phase")
    lineage = saved.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Staged checkpoint is missing concrete ancestry")
    parent, verified = load_stage_parent(
        lineage["parent_checkpoint"], lineage["parent_sha256"]
    )
    if lineage != verified:
        raise ValueError("Staged checkpoint lineage differs from verified parent")
    validate_parent_compatibility(
        parent, recipe,
        temporal_order=recipe.get("input", {}).get("temporal_order", "source"),
    )
    if recipe.get("parent_checkpoint") != {
        "global_step": parent["global_step"],
        "recipe": scientific_parent_recipe(parent["recipe"]),
    }:
        raise ValueError("Scientific Phase 1 recipe differs from the actual ancestor")
    if saved.get("total_encoder_updates") != parent["global_step"] + saved["global_step"]:
        raise ValueError("Staged checkpoint cumulative update count mismatch")
    return lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        raise ValueError("Unexpected upstream revision")
    if args.output.exists():
        raise FileExistsError(args.output)
    # These are trusted, locally produced full-state checkpoints. Python/NumPy
    # RNG and optimizer state require the full pickle loader.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("kind") != "olmoearth_alignment" or saved.get("format_version") != 1:
        raise ValueError("Expected an alignment full-state checkpoint")
    recipe = saved["recipe"]
    config = saved["encoder_config"]
    if config != recipe["encoder_config"]:
        raise ValueError("Encoder config differs from the declared training recipe")
    lineage = validate_checkpoint_lineage(saved)
    if recipe.get("upstream") != revision:
        raise ValueError("Checkpoint upstream differs from the export runtime")
    adapter = Path(__file__).with_name("infer.py")
    if sha256(adapter) != recipe.get("source_hashes", {}).get("infer"):
        raise ValueError("Inference adapter differs from the recorded training recipe")
    sys.path.insert(0, str(args.upstream.resolve()))
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

    encoder = EncoderConfig.from_dict(config).build()
    encoder.load_state_dict(saved["encoder"], strict=True)
    if not all(
        torch.isfinite(tensor).all() for tensor in encoder.state_dict().values()
    ):
        raise ValueError("Nonfinite encoder state")
    args.output.mkdir(parents=True)
    torch.save(saved["encoder"], args.output / "encoder.pt")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(adapter, args.output / "infer.py")
    shutil.copyfile(args.upstream / "LICENSE", args.output / "LICENSE-OlmoEarth")
    provenance = {
        "upstream": revision,
        "origin": "own_from_scratch_training",
        "variant": "v1/" + recipe["model_variant"],
        "inference_adapter_version": 3,
        "infer_sha256": sha256(adapter),
        "initialization_seed": recipe["init_seed"],
        "data_seed": recipe["data_seed"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": saved["global_step"],
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "training_recipe": confirmation_recipe(recipe, saved["global_step"]),
        "training_recipe_source": str(args.checkpoint.resolve()),
        "training_implementation": {
            "name": "train_alignment",
            "source_hashes": recipe.get("source_hashes"),
        },
        "training_validity": "not_flagged_invalid",
        "planned_training_duration": recipe.get("schedule"),
        "encoder_sha256": sha256(args.output / "encoder.pt"),
    }
    if lineage is not None:
        provenance["training_lineage"] = lineage
        provenance["total_encoder_updates"] = saved["total_encoder_updates"]
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()

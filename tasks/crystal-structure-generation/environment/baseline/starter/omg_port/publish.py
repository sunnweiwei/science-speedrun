#!/usr/bin/env python3
"""Publish a relocatable, inference-only CrystalGen OMatG package."""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


MP20_TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"


def swap_into(staged: Path, final: Path) -> None:
    """Replace ``final`` by ``staged`` with one rename, so a reader never sees a partial package."""
    old = final.with_name(f".{final.name}.old-{os.getpid()}")
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move its
            # contents aside instead; the swap below is still a single rename.
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--sampler-stats", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.sampler_stats.is_file():
        raise FileNotFoundError("checkpoint and sampler stats must exist")
    stats = json.loads(args.sampler_stats.read_text())
    if stats.get("schema") != "crystalgen-atom-count-prior/v0":
        raise ValueError("invalid sampler statistics")

    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("expected a Lightning checkpoint with state_dict")
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in checkpoint["state_dict"].items()
        if torch.is_tensor(value)
    }
    if not state:
        raise ValueError("checkpoint has no tensors")
    parameter_count = sum(tensor.numel() for tensor in state.values())
    if not 1 <= parameter_count <= 70_000_000:
        raise ValueError(f"checkpoint has {parameter_count} tensor values; limit is 70M")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".crystalgen-omg-", dir=args.output.parent))
    source_root = Path(__file__).resolve().parent
    try:
        model_path = temporary / "model.safetensors"
        save_file(state, model_path)
        shutil.copyfile(args.sampler_stats, temporary / "sampler_stats.json")
        shutil.copyfile(source_root / "config.yaml", temporary / "config.yaml")
        shutil.copyfile(source_root / "package_generate.sh", temporary / "generate.sh")
        package = temporary / "omg_port"
        package.mkdir()
        for name in ("__init__.py", "runtime.py", "generation.py"):
            shutil.copyfile(source_root / name, package / name)
        (temporary / "README.md").write_text(
            "# CrystalGen unconditional OMatG submission\n\n"
            "One from-scratch OMatG parameter set for unconditional crystal generation.\n"
        )
        manifest = {
            "schema": "crystalgen-model/v0",
            "checkpoint": {
                "path": "model.safetensors",
                "sha256": sha256(model_path),
                "format": "safetensors",
            },
            "parameter_count": parameter_count,
            "one_parameter_set": True,
            "trained_from_scratch": True,
            "training_data": {
                "name": "MP-20",
                "split": "train",
                "source_sha256": MP20_TRAIN_SHA256,
            },
            "model_config": {
                "architecture": "OMatG-DNG-unconditional",
                "omg_revision": "ed80cb86fd69d4f4d14164c6635b0a8d4939b34f",
                "generation_mode": "unconditional",
            },
        }
        (temporary / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.chmod(temporary / "generate.sh", 0o755)
        swap_into(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()

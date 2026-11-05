#!/usr/bin/env python3
"""Evaluate an exported reference model on BioLiP dev without training."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline/drugclip"))

import torch

from common import contained, sha256
from data import POCKET_CROP_POLICY, PairedDataset
from predict import load_model
from train import evaluate, make_validation_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-records", type=int, help="smoke check only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.validation_batch_size < 2 or args.workers < 0:
        parser.error("batch size must be at least two and workers nonnegative")
    if args.max_records is not None and args.max_records < 2:
        parser.error("max-records must be at least two")
    package = args.package.resolve()
    model, manifest = load_model(package, torch.device(args.device))
    summary = json.loads(contained(package, manifest["training_summary"]).read_text())
    crop = summary.get("pocket_crop", {})
    if crop.get("policy") != POCKET_CROP_POLICY:
        raise ValueError("package has no compatible pocket crop recipe")
    dev = PairedDataset(
        args.input_root / "processed/biolip_dev/data.lmdb",
        max_molecule_atoms=model.config.max_molecule_atoms,
        max_pocket_atoms=model.config.max_pocket_atoms,
        seed=int(crop["seed"]), max_records=args.max_records,
        unimol_molecule=True, unimol_pocket=True,
    )
    if len(dev) < 2:
        raise ValueError("development retrieval requires at least two records")
    loader = make_validation_loader(dev, args, rank=0, world=1)
    result = {
        "schema": "drugclip-biolip-dev/v1",
        "metrics": evaluate(model, loader, torch.device(args.device), world=1),
        "records": len(dev), "validation_batch_size": args.validation_batch_size,
        "world_size": 1, "epoch": 0, "pocket_crop": crop,
        "smoke_limit_requested": args.max_records is not None,
        "model_manifest_sha256": sha256(package / "model_manifest.json"),
        "data_manifest_sha256": sha256(args.input_root / "processed/manifest.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

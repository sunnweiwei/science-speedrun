#!/usr/bin/env python3
"""Run streaming inference with a published starter model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import contained, sha256
from data import SingleStructureDataset
from model import ModelConfig, StarterModel


def test_targets(root: Path):
    for benchmark, relative, pocket_name in (
        ("dude", "test/dude", "pocket.lmdb"),
        ("lit_pcba", "test/lit_pcba", "pockets.lmdb"),
    ):
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for target in sorted(path for path in directory.iterdir() if path.is_dir()):
            molecules, pockets = target / "mols.lmdb", target / pocket_name
            if benchmark == "lit_pcba" and not pockets.is_file():
                pockets = target / "pocket.lmdb"
            if not molecules.is_file() or not pockets.is_file():
                raise FileNotFoundError(f"missing benchmark LMDB under {target}")
            yield benchmark, target.name, molecules, pockets


def load_model(package: Path, device: torch.device) -> StarterModel:
    manifest = json.loads((package / "model_manifest.json").read_text())
    if manifest.get("schema") != "target-based-drug-screening-model/v1":
        raise ValueError("unsupported model manifest")
    checkpoint = contained(package, manifest["checkpoint"]["path"])
    if sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    declared = json.loads(contained(package, manifest["model_config"]).read_text())
    if saved["model_config"] != declared:
        raise ValueError("checkpoint and model config disagree")
    model = StarterModel(ModelConfig(**declared))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def encode(model, path: Path, kind: str, batch_size: int, workers: int,
           device: torch.device) -> torch.Tensor:
    loader = DataLoader(SingleStructureDataset(path, kind),
                        batch_size=batch_size, shuffle=False, num_workers=workers)
    values = []
    for _, features in loader:
        function = model.encode_molecule if kind == "molecule" else model.encode_pocket
        values.append(function(features.to(device)))
    if not values:
        raise ValueError(f"empty LMDB: {path}")
    return torch.cat(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", choices=("all", "dude", "lit_pcba"), default="all")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-ligands", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    if ((args.max_targets is not None and args.max_targets < 1)
            or (args.max_ligands is not None and args.max_ligands < 1)):
        parser.error("smoke limits must be positive")
    root, output = args.input_root.resolve(), args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must start empty")
    device = torch.device(args.device)
    package = args.model_package.resolve()
    model = load_model(package, device)
    targets = [item for item in test_targets(root)
               if args.benchmark == "all" or item[0] == args.benchmark]
    if args.max_targets is not None:
        targets = targets[:args.max_targets]
    temporary, rows = output / "scores.csv.tmp", 0
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("benchmark", "target_id", "ligand_index", "score"))
        for benchmark, target, molecule_path, pocket_path in targets:
            pocket = encode(model, pocket_path, "pocket", args.batch_size,
                            args.workers, device)
            molecule_loader = DataLoader(
                SingleStructureDataset(molecule_path, "molecule", args.max_ligands),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            )
            for indices, features in molecule_loader:
                molecules = model.encode_molecule(features.to(device))
                scores = (pocket @ molecules.T).amax(dim=0).cpu().tolist()
                for index, score in zip(indices.tolist(), scores, strict=True):
                    writer.writerow((benchmark, target, index, repr(float(score))))
                    rows += 1
    os.replace(temporary, output / "scores.csv")
    receipt = {
        "schema": "target-based-drug-screening-predictions/v1",
        "model_manifest_sha256": sha256(package / "model_manifest.json"),
        "rows": rows, "targets": len(targets), "benchmark": args.benchmark,
        "complete_benchmark": args.max_targets is None and args.max_ligands is None,
    }
    (output / "prediction_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

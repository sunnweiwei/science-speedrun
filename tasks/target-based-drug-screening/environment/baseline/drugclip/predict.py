#!/usr/bin/env python3
"""Stream scores for every ligand in complete DUD-E and LIT-PCBA targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import contained, sha256
from data import SingleStructureDataset, collate_structures, move_features
from model import PaperConfig, PaperDrugCLIP


def load_model(package: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    manifest = json.loads((package / "model_manifest.json").read_text())
    if manifest.get("schema") != "target-based-drug-screening-model/v2":
        raise ValueError("unsupported model manifest")
    if manifest.get("method") != "paper-dual-unimol":
        raise ValueError("this reference requires a paper-scale dual Uni-Mol checkpoint")
    checkpoint_path = contained(package, manifest["checkpoint"]["path"])
    if sha256(checkpoint_path) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    declared = json.loads(contained(package, manifest["model_config"]).read_text())
    if checkpoint["model_config"] != declared:
        raise ValueError("checkpoint and declared model config disagree")
    model = PaperDrugCLIP(PaperConfig(**declared))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), manifest


def test_targets(data_root: Path, benchmark_filter: str = "all"):
    for benchmark, directory, pocket_name in (
        ("dude", data_root / "test/dude", "pocket.lmdb"),
        ("lit_pcba", data_root / "test/lit_pcba", "pockets.lmdb"),
    ):
        if benchmark_filter != "all" and benchmark != benchmark_filter:
            continue
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for target in sorted(path for path in directory.iterdir() if path.is_dir()):
            molecule_path = target / "mols.lmdb"
            pocket_path = target / pocket_name
            if not pocket_path.is_file() and benchmark == "lit_pcba":
                # A few compatible releases use the singular spelling.
                pocket_path = target / "pocket.lmdb"
            if not molecule_path.is_file() or not pocket_path.is_file():
                raise FileNotFoundError(f"missing benchmark LMDB under {target}")
            yield benchmark, target.name, molecule_path, pocket_path


@torch.no_grad()
def encode_pockets(model, path: Path, config, args, device):
    # The author's test.sh uses 511 pocket atoms even though training uses 256.
    dataset = SingleStructureDataset(path, "pocket", max_atoms=511,
                                     unimol_pocket=True, seed=args.pocket_crop_seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_structures,
                        pin_memory=device.type == "cuda")
    encoded = []
    for batch in loader:
        encoded.append(model.encode_pocket(move_features(batch["features"], device)))
    if not encoded:
        raise ValueError(f"empty pocket LMDB: {path}")
    return torch.cat(encoded)


def author_pool_scores(matrix: np.ndarray, benchmark: str) -> np.ndarray:
    """Match test_dude_target/test_pcba_target score pooling."""
    if benchmark == "dude":
        medians = np.median(matrix, axis=1, keepdims=True)
        mads = np.median(np.abs(matrix - medians), axis=1, keepdims=True)
        matrix = 0.6745 * (matrix - medians) / (mads + 1e-6)
    return matrix.max(axis=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", choices=("all", "dude", "lit_pcba"),
                        default="all")
    parser.add_argument("--max-targets", type=int,
                        help="smoke-test only; never use for official evaluation")
    parser.add_argument("--max-ligands", type=int,
                        help="smoke-test only; never use for official evaluation")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    if ((args.max_targets is not None and args.max_targets < 1)
            or (args.max_ligands is not None and args.max_ligands < 1)):
        parser.error("smoke limits must be positive")
    data_root = args.input_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must start empty")
    device = torch.device(args.device)
    package = args.model_package.resolve()
    model, manifest = load_model(package, device)
    summary = json.loads(contained(package, manifest["training_summary"]).read_text())
    crop = summary.get("pocket_crop", {})
    if crop.get("policy") != "author_distance_weighted_v1":
        raise ValueError("model package has no compatible pocket crop recipe")
    args.pocket_crop_seed = int(crop["seed"])
    config = model.config
    selected = list(test_targets(data_root, args.benchmark))
    if args.max_targets is not None:
        selected = selected[:args.max_targets]
    temporary = output / "scores.csv.tmp"
    row_count = 0
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "benchmark", "target_id", "ligand_index", "score"
        ))
        writer.writeheader()
        for benchmark, target_id, molecule_path, pocket_path in selected:
            pocket_embeddings = encode_pockets(model, pocket_path, config, args, device)
            molecules = SingleStructureDataset(
                molecule_path, "molecule", max_atoms=config.max_molecule_atoms,
                max_records=args.max_ligands,
                unimol_molecule=True,
            )
            molecule_loader = DataLoader(
                molecules, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, collate_fn=collate_structures,
                pin_memory=device.type == "cuda",
            )
            target_chunks = []
            for batch in molecule_loader:
                molecule_embeddings = model.encode_molecule(
                    move_features(batch["features"], device)
                )
                target_chunks.append((pocket_embeddings @ molecule_embeddings.T)
                                     .float().cpu().numpy())
            scores = author_pool_scores(np.concatenate(target_chunks, axis=1), benchmark)
            for index, score in enumerate(scores):
                writer.writerow({"benchmark": benchmark, "target_id": target_id,
                                 "ligand_index": index, "score": repr(float(score))})
                row_count += 1
    os.replace(temporary, output / "scores.csv")
    receipt = {
        "schema": "target-based-drug-screening-predictions/v1",
        "model_manifest_sha256": sha256(package / "model_manifest.json"),
        "rows": row_count, "targets": len(selected),
        "benchmark": args.benchmark,
        "complete_benchmark": args.max_targets is None and args.max_ligands is None,
        "pocket_crop": {**crop, "test_epoch": 0,
                        "test_max_atoms": 511},
    }
    (output / "prediction_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

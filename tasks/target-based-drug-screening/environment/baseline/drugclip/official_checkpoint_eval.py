#!/usr/bin/env python3
"""Evaluate released DrugCLIP benchmark checkpoints on frozen test LMDBs.

The score-producing tensors are loaded strictly into the parity-tested Uni-Mol
implementation in this directory. Auxiliary heads unused by the author's
test_dude_target/test_pcba_target are explicitly excluded. DUD-E uses the
author's per-pocket median/MAD normalization before max pooling; LIT-PCBA
uses raw cosine similarities before max pooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import sha256
from data import LMDBRecords, collate_structures, move_features, structure_features
from model import PaperConfig, PaperDrugCLIP


EXCLUDED_PREFIXES = (
    "classification_head.", "cross_distance_project.",
    "fuse_project.", "holo_distance_project.",
)
RENAMES = (
    ("mol_model.", "molecule_encoder."),
    ("pocket_model.", "pocket_encoder."),
    ("mol_project.linear1.", "molecule_projection.layers.0."),
    ("mol_project.linear2.", "molecule_projection.layers.2."),
    ("pocket_project.linear1.", "pocket_projection.layers.0."),
    ("pocket_project.linear2.", "pocket_projection.layers.2."),
)


def load_official_model(checkpoint: Path, device: torch.device) -> PaperDrugCLIP:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["model"]
    model = PaperDrugCLIP(PaperConfig(max_pocket_atoms=511,
                                      gradient_checkpointing=False))
    expected = model.state_dict()
    mapped = {}
    excluded = []
    for key, tensor in state.items():
        if key.startswith(EXCLUDED_PREFIXES):
            excluded.append(key)
            continue
        if key == "logit_scale":
            # Screening uses only normalized embeddings; scale is immaterial.
            mapped[key] = tensor.reshape(())
            continue
        for source, target in RENAMES:
            if key.startswith(source):
                key = target + key[len(source):]
                break
        mapped[key] = tensor
    if set(mapped) != set(expected):
        raise ValueError(f"checkpoint keys differ: missing={sorted(set(expected)-set(mapped))[:8]}, "
                         f"unexpected={sorted(set(mapped)-set(expected))[:8]}")
    wrong = [key for key in mapped if mapped[key].shape != expected[key].shape]
    if wrong or len(excluded) != 24:
        raise ValueError(f"checkpoint shape/auxiliary heads differ: {wrong[:8]}, "
                         f"excluded={len(excluded)}")
    model.load_state_dict(mapped, strict=True)
    return model.to(device).eval()


class OfficialTestStructures(Dataset):
    def __init__(self, path: Path, kind: str, max_atoms: int):
        self.records = LMDBRecords(path)
        self.kind = kind
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # The released test loop does not set an epoch, while Uni-Core's
        # numpy_seed requires an integer. Epoch 0 is the fixed evaluation
        # convention used here; most benchmark records have one conformer.
        mixed_seed = int(hash((1, 0, int(index))) % 1e8)
        if self.kind == "molecule":
            coordinates = record["coordinates"]
            choices = np.asarray(coordinates)
            conformer = (np.random.RandomState(mixed_seed).randint(len(choices))
                         if choices.ndim == 3 else 0)
            features = structure_features(
                record["atoms"], coordinates, max_atoms=self.max_atoms,
                seed=conformer, unimol=True,
            )
        else:
            features = structure_features(
                record["pocket_atoms"], record["pocket_coordinates"],
                max_atoms=self.max_atoms, seed=0, unimol_pocket=True,
                crop_policy="pocket_distance_weighted", crop_seed=1,
                epoch=0, index=index,
            )
        return {"features": features, "index": index, "name": "",
                "label": int(record.get("label", -1))}


def loader(path: Path, kind: str, max_atoms: int, batch_size: int, workers: int):
    return DataLoader(OfficialTestStructures(path, kind, max_atoms),
                      batch_size=batch_size, num_workers=workers,
                      collate_fn=collate_structures, shuffle=False)


@torch.inference_mode()
def score_target(model, benchmark: str, target: Path, batch_size: int,
                 workers: int, device: torch.device) -> np.ndarray:
    pocket_file = target / ("pocket.lmdb" if benchmark == "dude" else "pockets.lmdb")
    pockets = []
    for batch in loader(pocket_file, "pocket", 511, 1, workers):
        pockets.append(model.encode_pocket(move_features(batch["features"], device)))
    pocket_vectors = torch.cat(pockets)
    chunks = []
    for batch in loader(target / "mols.lmdb", "molecule", 256,
                        batch_size, workers):
        vectors = model.encode_molecule(move_features(batch["features"], device))
        chunks.append((pocket_vectors @ vectors.T).float().cpu().numpy())
    matrix = np.concatenate(chunks, axis=1)
    if benchmark == "dude":
        medians = np.median(matrix, axis=1, keepdims=True)
        mads = np.median(np.abs(matrix - medians), axis=1, keepdims=True)
        matrix = 0.6745 * (matrix - medians) / (mads + 1e-6)
    return matrix.max(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dude-checkpoint", type=Path, required=True)
    parser.add_argument("--pcba-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="limit for a wiring smoke only")
    parser.add_argument("--only-target", help="one benchmark/target for a focused audit")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipt = {"source_revision": "9d0db73915219763b5bcc3b7de86d7c4edb5ed03",
               "max_pocket_atoms": 511, "seed": 1, "test_epoch": 0,
               "checkpoints": {}, "targets": {}}
    device = torch.device(args.device)
    done = 0
    for benchmark, checkpoint in (("dude", args.dude_checkpoint),
                                  ("lit_pcba", args.pcba_checkpoint)):
        receipt["checkpoints"][benchmark] = {"path": str(checkpoint),
                                              "sha256": sha256(checkpoint)}
        model = load_official_model(checkpoint, device)
        for target in sorted((args.data_root / "test" / benchmark).iterdir()):
            if not target.is_dir():
                continue
            if args.only_target and f"{benchmark}/{target.name}" != args.only_target:
                continue
            if args.max_targets is not None and done >= args.max_targets:
                break
            destination = args.output_root / benchmark / f"{target.name}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                receipt["targets"][f"{benchmark}/{target.name}"] = "existing"
                done += 1
                continue
            values = score_target(model, benchmark, target, args.batch_size,
                                  args.workers, device)
            temporary = destination.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("benchmark", "target_id", "ligand_index", "score"))
                for index, score in enumerate(values):
                    writer.writerow((benchmark, target.name, index, repr(float(score))))
            os.replace(temporary, destination)
            receipt["targets"][f"{benchmark}/{target.name}"] = len(values)
            (args.output_root / "run_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(f"{benchmark}/{target.name}: {len(values)}", flush=True)
            done += 1
        del model
        torch.cuda.empty_cache()
    (args.output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.max_targets is None:
        merged = args.output_root / "scores.csv"
        with merged.open("w", newline="") as output:
            output.write("benchmark,target_id,ligand_index,score\n")
            for file in sorted(args.output_root.glob("*/*.csv")):
                with file.open() as source:
                    next(source)
                    for line in source:
                        output.write(line)


if __name__ == "__main__":
    main()

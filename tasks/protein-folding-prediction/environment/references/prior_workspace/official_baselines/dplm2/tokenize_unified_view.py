#!/usr/bin/env python3
"""Tokenize unified protein coordinates with the frozen official tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKBONE_ATOMS = {"N": 0, "CA": 1, "C": 2}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--struct-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_tokenizer(path: Path, device):
    import torch
    from byprot.models.structok.structok_lfq import VQModel
    from byprot.utils.config import load_yaml_config

    config = load_yaml_config(str(path / "config.yaml"))
    model = VQModel(**config)
    state = torch.load(path / "dplm2_struct_tokenizer.ckpt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise AssertionError(
            f"structure-tokenizer state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.requires_grad_(False).eval().to(device)


def coordinates(record: dict):
    import numpy as np

    data = np.load(record["structure_path"], allow_pickle=False)
    chain = str(record["chain_id"])
    chain_mask = data["chain_id"].astype(str) == chain
    length = len(record["aa_seq"])
    result = np.full((length, 3, 3), np.nan, dtype=np.float32)
    for coordinate, residue_id, atom_name in zip(
        data["coord"][chain_mask],
        data["res_id"][chain_mask],
        data["atom_name"][chain_mask],
        strict=True,
    ):
        atom_index = BACKBONE_ATOMS.get(str(atom_name))
        residue_index = int(residue_id) - 1
        if atom_index is not None and 0 <= residue_index < length:
            result[residue_index, atom_index] = coordinate
    valid = np.isfinite(result).all(axis=(1, 2))
    result[~valid] = 0.0
    return result, valid


def batches(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = arguments()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("structure tokenization requires an explicitly selected GPU")
    records = [
        json.loads(line)
        for index, line in enumerate(
            (args.staging / f"{args.split}.jsonl").open()
        )
        if index % args.world_size == args.rank
    ]
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = load_tokenizer(args.struct_tokenizer, torch.device("cuda"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_residues = 0
    with args.output.open("w") as destination, torch.inference_mode():
        for group in batches(records, args.batch_size):
            unpacked = [coordinates(record) for record in group]
            maximum = max(len(value[0]) for value in unpacked)
            atom_positions = torch.zeros(
                len(group), maximum, 3, 3, dtype=torch.float32, device="cuda"
            )
            residue_mask = torch.zeros(
                len(group), maximum, dtype=torch.float32, device="cuda"
            )
            lengths = torch.tensor(
                [len(value[0]) for value in unpacked], device="cuda"
            )
            for index, (position, valid) in enumerate(unpacked):
                length = len(position)
                atom_positions[index, :length] = torch.from_numpy(position).cuda()
                residue_mask[index, :length] = torch.from_numpy(valid).cuda()
                missing_residues += int((~valid).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = tokenizer.tokenize(
                    atom_positions=atom_positions,
                    res_mask=residue_mask,
                    seq_length=lengths,
                )
            tokens = tokens.cpu().numpy()
            masks = residue_mask.bool().cpu().numpy()
            for index, record in enumerate(group):
                length = len(record["aa_seq"])
                struct_seq = [
                    f"{int(token):04d}" if masks[index, offset] else "<unk_struct>"
                    for offset, token in enumerate(tokens[index, :length])
                ]
                output = {
                    "aa_seq": record["aa_seq"],
                    "struct_seq": ",".join(struct_seq),
                    "length": length,
                    "pdb_name": record["pdb_name"],
                }
                destination.write(json.dumps(output, sort_keys=True) + "\n")
                written += 1
    report = {
        "schema_version": "foldingtrain-dplm2-structure-token-shard-v1",
        "split": args.split,
        "rank": args.rank,
        "world_size": args.world_size,
        "records": written,
        "missing_backbone_residues": missing_residues,
        "official_structure_tokenizer": str(args.struct_tokenizer),
        "quality_comparable": args.limit is None,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

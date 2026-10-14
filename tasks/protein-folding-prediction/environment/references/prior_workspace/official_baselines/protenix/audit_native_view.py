#!/usr/bin/env python3
"""Prove that Protenix native indices contain only frozen-train structures."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from official_baselines.data_policy import (
    UNIFIED_DATASET,
    UNIFIED_SPLIT_SCHEMA,
    UNIFIED_TRAIN_SHA256,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-train-split", type=Path, required=True)
    parser.add_argument("--native-index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def main() -> None:
    args = arguments()
    split_digest = sha256(args.unified_train_split)
    if split_digest != UNIFIED_TRAIN_SHA256:
        raise ValueError(
            f"unified train split digest {split_digest} != {UNIFIED_TRAIN_SHA256}"
        )
    common_pdb_ids = set()
    records = 0
    with args.unified_train_split.open() as stream:
        for line in stream:
            record = json.loads(line)
            records += 1
            if record.get("unified_split") != "train":
                raise ValueError(f"non-train record in frozen train split: {record}")
            if record.get("source") == "openfold3_pdb":
                common_pdb_ids.add(
                    str(record.get("pdb_id", record["source_id"])).lower()
                )

    indices = []
    all_native_ids = set()
    for path in args.native_index:
        with open_csv(path) as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "pdb_id" not in rows[0]:
            raise ValueError(f"native index has no pdb_id rows: {path}")
        pdb_ids = {row["pdb_id"].lower() for row in rows}
        outside = sorted(pdb_ids - common_pdb_ids)
        if outside:
            raise ValueError(
                f"native index contains {len(outside)} structures outside frozen train: "
                f"{outside[:10]}"
            )
        all_native_ids.update(pdb_ids)
        indices.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(rows),
                "pdb_ids": sorted(pdb_ids),
            }
        )
    report = {
        "schema_version": "foldingtrain-protenix-native-view-v1",
        "dataset": UNIFIED_DATASET,
        "source_split_schema": UNIFIED_SPLIT_SCHEMA,
        "source_train_split": str(args.unified_train_split),
        "source_train_sha256": split_digest,
        "source_train_records": records,
        "source_train_pdb_structures": len(common_pdb_ids),
        "native_indices": indices,
        "native_pdb_structures": len(all_native_ids),
        "outside_frozen_train": 0,
        "data_policy": "deterministic native views only; no upstream data substitution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

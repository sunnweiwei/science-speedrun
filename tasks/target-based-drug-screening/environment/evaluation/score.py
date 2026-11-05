#!/usr/bin/env python3
"""Evaluate streamed predictions with the metrics used by DrugCLIP."""

from __future__ import annotations

import argparse
import csv
from itertools import groupby
import json
from pathlib import Path
import pickle

import numpy as np
from metrics import ranking_metrics
from aggregation import summarize_targets

from input_view import discover_targets, load_permutation



class LMDBRecords:
    """Minimal evaluator-owned LMDB label reader."""

    def __init__(self, path: Path) -> None:
        import lmdb

        self.path = path
        self.environment = lmdb.open(
            str(path), subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        with self.environment.begin() as transaction:
            self.count = transaction.stat()["entries"]
            if self.count and (transaction.get(b"0") is None
                               or transaction.get(str(self.count - 1).encode()) is None):
                raise ValueError(f"LMDB does not use contiguous numeric keys: {path}")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        with self.environment.begin(buffers=True) as transaction:
            value = transaction.get(str(index).encode("ascii"))
            if value is None:
                raise RuntimeError(f"missing LMDB record {index}: {self.path}")
            return pickle.loads(bytes(value))

    def close(self):
        self.environment.close()



def lmdb_path(data_root: Path, benchmark: str, target: str) -> Path:
    return data_root / "test" / benchmark / target / "mols.lmdb"


def score_target(rows, data_root: Path, private_mapping: Path | None = None) -> dict:
    rows = list(rows)
    benchmark, target = rows[0]["benchmark"], rows[0]["target_id"]
    records = LMDBRecords(lmdb_path(data_root, benchmark, target))
    values = []
    try:
        if len(rows) != len(records):
            raise ValueError(f"{benchmark}/{target}: expected {len(records)} scores, got {len(rows)}")
        observed = [int(row["ligand_index"]) for row in rows]
        if observed != list(range(len(records))):
            raise ValueError(f"{benchmark}/{target}: ligand indices are missing or reordered")
        order = (load_permutation(private_mapping, benchmark, target, len(records))
                 if private_mapping is not None else range(len(records)))
        for index, row in enumerate(rows):
            label = int(records[int(order[index])].get("label", -1))
            score = float(row["score"])
            if label not in (0, 1) or not np.isfinite(score):
                raise ValueError(f"{benchmark}/{target}: invalid label or score at {index}")
            values.append([score, label])
    finally:
        records.close()
    values.sort(key=lambda item: item[0], reverse=True)
    active = sum(item[1] for item in values)
    if not 0 < active < len(values):
        raise ValueError(f"{benchmark}/{target}: metric requires both classes")
    metrics = ranking_metrics(values)
    return {
        "benchmark": benchmark, "target_id": target,
        "ligands": len(values), "actives": active,
        **metrics,
    }


def evaluate(data_root: Path, predictions: Path,
             private_mapping: Path | None = None, *, require_full_suite=False) -> dict:
    expected_targets = discover_targets(data_root)
    seen = set()
    per_target = []
    with predictions.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["benchmark", "target_id", "ligand_index", "score"]
        if reader.fieldnames != expected:
            raise ValueError(f"score columns must be {expected}")
        for key, group in groupby(reader, key=lambda row: (row["benchmark"], row["target_id"])):
            if key not in expected_targets or key in seen:
                raise ValueError(f"unexpected or repeated target: {key}")
            seen.add(key)
            per_target.append(score_target(group, data_root.resolve(), private_mapping))
    if seen != set(expected_targets):
        raise ValueError(f"missing targets: {sorted(set(expected_targets) - seen)}")
    if not per_target:
        raise ValueError("prediction file contains no targets")
    return {
        **summarize_targets(per_target, require_full_suite=require_full_suite),
        "per_target": per_target,
        "tie_policy": "expected metric over permutations of exactly tied scores",
        "input_order": "private per-target permutation" if private_mapping else "source order",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--private-mapping", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data_root, args.predictions, args.private_mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()


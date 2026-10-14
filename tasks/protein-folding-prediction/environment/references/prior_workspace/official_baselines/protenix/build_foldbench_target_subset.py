#!/usr/bin/env python3
"""Materialize target CSVs matching a coordinate-free FoldBench sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.selection_manifest.read_text())
    selection = manifest["selection"]
    pdb_ids = {pdb_id.lower() for pdb_id in selection["pdb_ids"]}
    expected = selection["targets_by_track"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, int] = {}
    for track, expected_count in expected.items():
        source = args.targets_dir / f"{track}.csv"
        destination = args.output_dir / source.name
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["pdb_id"].lower() in pdb_ids]
            fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"target table has no header: {source}")
        if len(rows) != expected_count:
            raise ValueError(
                f"{track}: selected {len(rows)} rows, expected {expected_count}"
            )
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        observed[track] = len(rows)

    audit = {
        "schema": "foldbench-target-subset-v1",
        "selection_manifest": str(args.selection_manifest),
        "pdb_ids": sorted(pdb_ids),
        "targets_by_track": observed,
    }
    (args.output_dir / "subset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

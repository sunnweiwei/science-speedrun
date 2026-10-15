#!/usr/bin/env python3
"""Fit the portable atom-count prior used to seed OMatG generation."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from pymatgen.core import Structure

    counts: dict[str, int] = {}
    rows = 0
    with args.train_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if "cif" not in (reader.fieldnames or []):
            raise ValueError("training CSV is missing cif")
        for row in reader:
            n_atoms = len(Structure.from_str(row["cif"], fmt="cif"))
            if not 1 <= n_atoms <= 20:
                raise ValueError(f"MP-20 row has {n_atoms} atoms")
            counts[str(n_atoms)] = counts.get(str(n_atoms), 0) + 1
            rows += 1
    if rows != 27_136:
        raise ValueError(f"expected 27136 MP-20 training rows, got {rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "crystalgen-atom-count-prior/v0",
        "training_rows": rows,
        "counts": counts,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the split boundary of the experimental and synthetic MIST corpora."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def key2d(value: str) -> str:
    return value.strip().split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument(
        "--experimental-tsv",
        type=Path,
        help="MassSpecGym TSV containing identifier, inchikey, and fold.",
    )
    parser.add_argument("--synthetic-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-synthetic-rows", type=int, required=True)
    args = parser.parse_args()

    folds: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    counts = {name: 0 for name in folds}
    if args.experimental_tsv is not None:
        if args.labels is not None or args.split is not None:
            parser.error("--experimental-tsv cannot be combined with --labels/--split")
        seen_specs: set[str] = set()
        with args.experimental_tsv.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                name, fold = row["identifier"], row["fold"]
                if fold not in folds or name in seen_specs:
                    raise ValueError(f"invalid experimental row for {name!r}/{fold!r}")
                seen_specs.add(name)
                counts[fold] += 1
                folds[fold].add(key2d(row["inchikey"]))
    else:
        if args.labels is None or args.split is None:
            parser.error("supply --experimental-tsv or both --labels and --split")
        with args.labels.open(newline="", encoding="utf-8") as stream:
            experimental = {
                row["spec"]: key2d(row["inchikey"])
                for row in csv.DictReader(stream, delimiter="\t")
            }
        seen_specs = set()
        with args.split.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                name, fold = row["name"], row["split"]
                if fold not in folds or name not in experimental or name in seen_specs:
                    raise ValueError(f"invalid split row for {name!r}/{fold!r}")
                seen_specs.add(name)
                counts[fold] += 1
                folds[fold].add(experimental[name])
        if seen_specs != set(experimental):
            raise ValueError("split does not cover every experimental spectrum exactly once")

    pairwise_overlap = {
        f"{left}_{right}": len(folds[left] & folds[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(pairwise_overlap.values()):
        raise ValueError(f"experimental 2D structure overlap: {pairwise_overlap}")

    synthetic_rows = 0
    synthetic_keys: set[str] = set()
    with args.synthetic_labels.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"spec", "smiles", "inchikey", "formula"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"unexpected synthetic label fields: {reader.fieldnames}")
        for row in reader:
            synthetic_rows += 1
            synthetic_keys.add(key2d(row["inchikey"]))

    synthetic_overlap = {
        fold: len(synthetic_keys & folds[fold]) for fold in ("val", "test")
    }
    if any(synthetic_overlap.values()):
        raise ValueError(f"synthetic validation/test structure overlap: {synthetic_overlap}")

    expected_counts = {"train": 194119, "val": 19429, "test": 17556}
    if counts != expected_counts or synthetic_rows != args.expected_synthetic_rows:
        raise ValueError(
            f"unexpected corpus counts: folds={counts}, synthetic={synthetic_rows}"
        )
    report = {
        "schema": "massspecgym-retrieval-training-audit/v1",
        "experimental_spectra": sum(counts.values()),
        "experimental_fold_counts": counts,
        "experimental_unique_2d_structures": {
            fold: len(values) for fold, values in folds.items()
        },
        "experimental_pairwise_2d_overlap": pairwise_overlap,
        "synthetic_spectra": synthetic_rows,
        "synthetic_unique_2d_structures": len(synthetic_keys),
        "synthetic_overlap_with_held_out": synthetic_overlap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

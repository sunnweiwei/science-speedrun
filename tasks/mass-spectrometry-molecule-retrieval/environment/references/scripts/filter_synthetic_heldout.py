#!/usr/bin/env python3
"""Remove exact held-out 2D structures from packed synthetic MIST data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger


def key2d(value: str) -> str:
    return value.strip().split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument(
        "--experimental-tsv",
        type=Path,
        help="MassSpecGym TSV containing inchikey and fold; replaces --labels/--split.",
    )
    parser.add_argument("--synthetic-labels", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-labels", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    excluded_folds: dict[str, set[str]] = defaultdict(set)
    if args.experimental_tsv is not None:
        if args.labels is not None or args.split is not None:
            parser.error("--experimental-tsv cannot be combined with --labels/--split")
        with args.experimental_tsv.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["fold"] in {"val", "test"}:
                    excluded_folds[key2d(row["inchikey"])].add(row["fold"])
    else:
        if args.labels is None or args.split is None:
            parser.error("supply --experimental-tsv or both --labels and --split")
        with args.labels.open(newline="", encoding="utf-8") as stream:
            experimental = {
                row["spec"]: key2d(row["inchikey"])
                for row in csv.DictReader(stream, delimiter="\t")
            }
        with args.split.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["split"] in {"val", "test"}:
                    excluded_folds[experimental[row["name"]]].add(row["split"])

    temporary = args.output_labels.with_suffix(args.output_labels.suffix + ".tmp")
    removed_specs: list[str] = []
    removed_keys: Counter[str] = Counter()
    invalid_smiles: Counter[str] = Counter()
    smiles_validity: dict[str, bool] = {}
    total = kept = 0
    RDLogger.DisableLog("rdApp.*")
    with (
        args.synthetic_labels.open(newline="", encoding="utf-8") as source,
        temporary.open("w", newline="", encoding="utf-8") as target,
    ):
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("synthetic labels have no header")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            total += 1
            key = key2d(row["inchikey"])
            smiles = row["smiles"]
            valid_smiles = smiles_validity.get(smiles)
            if valid_smiles is None:
                valid_smiles = Chem.MolFromSmiles(smiles) is not None
                smiles_validity[smiles] = valid_smiles
            if key in excluded_folds:
                removed_specs.append(row["spec"])
                removed_keys[key] += 1
            elif not valid_smiles:
                removed_specs.append(row["spec"])
                invalid_smiles[smiles] += 1
            else:
                writer.writerow(row)
                kept += 1
    os.replace(temporary, args.output_labels)

    connection = sqlite3.connect(args.database)
    connection.executemany(
        "DELETE FROM subformulae WHERE spec = ?",
        ((spec,) for spec in removed_specs),
    )
    connection.commit()
    database_rows = connection.execute("SELECT COUNT(*) FROM subformulae").fetchone()[0]
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if database_rows != kept or integrity != "ok" or not removed_specs:
        raise ValueError(
            f"invalid filtered corpus: labels={kept}, database={database_rows}, "
            f"removed={len(removed_specs)}, integrity={integrity!r}"
        )

    report = {
        "schema": "massspecgym-retrieval-synthetic-filter/v1",
        "input_rows": total,
        "retained_rows": kept,
        "removed_rows": len(removed_specs),
        "removed_heldout_rows": sum(removed_keys.values()),
        "removed_invalid_smiles_rows": sum(invalid_smiles.values()),
        "removed_unique_2d_structures": len(removed_keys),
        "removed": [
            {
                "inchikey_2d": key,
                "spectra": removed_keys[key],
                "held_out_folds": sorted(excluded_folds[key]),
            }
            for key in sorted(removed_keys)
        ],
        "invalid_smiles": [
            {"smiles": smiles, "spectra": invalid_smiles[smiles]}
            for smiles in sorted(invalid_smiles)
        ],
        "database_integrity": integrity,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

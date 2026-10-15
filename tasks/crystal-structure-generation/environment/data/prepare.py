#!/usr/bin/env python3
"""Validate pinned MP-20 CSVs and add deterministic density labels."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


SOURCE_REVISION = "f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea"
SPLITS = {
    "train": (27136, "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"),
    "val": (9047, "7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29"),
    "test": (9046, "dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a"),
}
SOURCE_COLUMNS = {
    "material_id", "formation_energy_per_atom", "band_gap", "pretty_formula",
    "e_above_hull", "elements", "cif", "spacegroup.number",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path: Path) -> tuple[int, list[str]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        return sum(1 for _ in reader), header


def enrich(source: Path, destination: Path) -> int:
    try:
        from pymatgen.core import Structure
    except ImportError as error:
        raise RuntimeError("pymatgen is required to derive density labels") from error

    with source.open(newline="") as input_stream, destination.open("w", newline="") as output_stream:
        reader = csv.DictReader(input_stream)
        if not SOURCE_COLUMNS.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"unexpected MP-20 columns in {source}")
        fieldnames = list(reader.fieldnames or []) + ["density"]
        writer = csv.DictWriter(output_stream, fieldnames=fieldnames)
        writer.writeheader()
        count = 0
        for count, row in enumerate(reader, 1):
            structure = Structure.from_str(row["cif"], fmt="cif")
            row["density"] = format(float(structure.density), ".12g")
            writer.writerow(row)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output-root", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()

    source_report = {}
    for split, (expected_rows, expected_hash) in SPLITS.items():
        path = args.source_root / f"{split}.csv"
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"MP-20 {split} source checksum mismatch")
        rows, header = count_rows(path)
        if rows != expected_rows or not SOURCE_COLUMNS.issubset(set(header)):
            raise ValueError(f"MP-20 {split} shape mismatch")
        source_report[split] = {"rows": rows, "sha256": expected_hash}

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix=".crystalgen-", dir=args.output_root.parent))
    try:
        outputs = {}
        for split in ("train", "val", "test"):
            destination = temporary / f"{split}.csv"
            rows = enrich(args.source_root / f"{split}.csv", destination)
            outputs[split] = {"rows": rows, "sha256": sha256(destination)}
        manifest = {
            "schema": "crystalgen-mp20/v0",
            "source": {"repository": "txie-93/cdvae", "revision": SOURCE_REVISION},
            "source_splits": source_report,
            "prepared_splits": outputs,
            "derived_columns": {"density": "pymatgen Structure.density in g/cm^3"},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        for path in temporary.iterdir():
            os.replace(path, args.output_root / path.name)
    finally:
        temporary.rmdir()


if __name__ == "__main__":
    main()

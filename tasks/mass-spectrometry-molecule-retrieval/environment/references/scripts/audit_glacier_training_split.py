#!/usr/bin/env python3
"""Audit the public data boundary used by the released GLACIER recipe."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


FOLDS = ("train", "val", "test")
SUPPORTED_INSTRUMENTS = {"Orbitrap", "QTOF"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def released_domain(row: dict) -> bool:
    """Public-field predicate used by the task adapter."""

    return (
        row["adduct"] == "[M+H]+"
        and finite(row["collision_energy"])
        and row["instrument_type"] in SUPPORTED_INSTRUMENTS
    )


def read_upstream_specs(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames is None or "spec" not in rows.fieldnames:
            raise ValueError(f"{path} has no spec column")
        specs = {row["spec"] for row in rows}
    if "" in specs:
        raise ValueError(f"{path} contains an empty spec identifier")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tsv", required=True, type=Path)
    parser.add_argument("--upstream-labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    row_counts = Counter()
    public_domain_counts = Counter()
    simulation_counts = Counter()
    all_keys = {fold: set() for fold in FOLDS}
    public_domain_keys = {fold: set() for fold in FOLDS}
    public_domain_specs = set()
    simulation_specs = set()
    with args.source_tsv.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        required = {
            "identifier",
            "inchikey",
            "adduct",
            "instrument_type",
            "collision_energy",
            "fold",
            "simulation_challenge",
        }
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(f"{args.source_tsv} lacks required columns")
        for row in rows:
            fold = row["fold"]
            if fold not in all_keys:
                raise ValueError(f"unexpected fold {fold!r}")
            spec = row["identifier"]
            key_2d = row["inchikey"].split("-")[0]
            if not spec or not key_2d:
                raise ValueError("empty spectrum identifier or InChIKey")
            row_counts[fold] += 1
            all_keys[fold].add(key_2d)
            if released_domain(row):
                public_domain_counts[fold] += 1
                public_domain_keys[fold].add(key_2d)
                public_domain_specs.add(spec)
            if truthy(row["simulation_challenge"]):
                simulation_counts[fold] += 1
                simulation_specs.add(spec)

    if public_domain_specs != simulation_specs:
        raise ValueError(
            "public released-domain predicate does not reproduce "
            "simulation_challenge membership"
        )
    overlap = {
        f"{left}_{right}": len(all_keys[left] & all_keys[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"2D molecular identity crosses folds: {overlap}")

    upstream = None
    if args.upstream_labels is not None:
        upstream_specs = read_upstream_specs(args.upstream_labels)
        if upstream_specs != public_domain_specs:
            raise ValueError("upstream labels do not equal the released-domain spec set")
        upstream = {
            "path": str(args.upstream_labels.resolve()),
            "rows": len(upstream_specs),
            "sha256": sha256(args.upstream_labels),
            "exact_spec_set_match": True,
        }

    report = {
        "schema": "glacier-training-split-audit/v1",
        "source": {
            "path": str(args.source_tsv.resolve()),
            "sha256": sha256(args.source_tsv),
        },
        "rows": {fold: row_counts[fold] for fold in FOLDS},
        "released_domain_rows": {
            fold: public_domain_counts[fold] for fold in FOLDS
        },
        "simulation_challenge_rows": {
            fold: simulation_counts[fold] for fold in FOLDS
        },
        "unique_2d_structures": {fold: len(all_keys[fold]) for fold in FOLDS},
        "released_domain_unique_2d_structures": {
            fold: len(public_domain_keys[fold]) for fold in FOLDS
        },
        "cross_fold_2d_intersections": overlap,
        "public_predicate_exactly_matches_simulation_challenge": True,
        "upstream_labels": upstream,
        "scope_note": (
            "This verifies the public split and adapter domain. It cannot independently "
            "attest which rows were used to create a third-party released checkpoint."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

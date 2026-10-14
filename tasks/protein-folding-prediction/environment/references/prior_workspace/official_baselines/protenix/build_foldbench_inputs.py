#!/usr/bin/env python3
"""Convert the coordinate-free FoldBench index to official Protenix inputs.

The output schema is the schema consumed by the upstream Protenix inference
runner and is equivalent to FoldBench's official Protenix ``preprocess.py``
output.  This program never reads FoldBench ground-truth structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable


POLYMER_KEYS = {
    "protein": "proteinChain",
    "rna": "rnaSequence",
    "dna": "dnaSequence",
}
STANDARD_ALPHABETS = {
    "protein": set("ACDEFGHIKLMNPQRSTVWY"),
    "rna": set("ACGU"),
    "dna": set("ACGT"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assembly_count(
    record: dict[str, Any], entity_id: str, asym_ids: list[str]
) -> int:
    """Infer assembly-generated copies from FoldBench's ``A-2`` chain IDs."""
    copies_by_base = {asym_id: 1 for asym_id in asym_ids}
    for target in record["target_rows"]:
        for reference in target["referenced_chains"]:
            if reference["entity_id"] != entity_id:
                continue
            base = reference["base_asym_id"]
            chain_id = reference["chain_id"]
            if base not in copies_by_base:
                copies_by_base[base] = 1
            match = re.fullmatch(re.escape(base) + r"-(\d+)", chain_id)
            copy_number = int(match.group(1)) if match else 1
            copies_by_base[base] = max(copies_by_base[base], copy_number)
    return sum(copies_by_base.values())


def estimated_tokens(record: dict[str, Any]) -> int:
    polymer_tokens = sum(
        len(polymer["sequence"])
        * assembly_count(
            record, polymer["entity_id"], polymer["asym_ids"]
        )
        for polymer in record["polymers"]
    )
    # Ligand atom counts are unavailable in the coordinate-free index.  This
    # small deterministic penalty only guides smoke-sample selection.
    return polymer_tokens + 32 * sum(
        assembly_count(record, nonpolymer["entity_id"], nonpolymer["asym_ids"])
        for nonpolymer in record["nonpolymers"]
    )


def select_per_track(
    records: Iterable[dict[str, Any]], per_track: int
) -> list[dict[str, Any]]:
    """Select short deterministic assemblies while covering every track."""
    records = [
        record
        for record in records
        if all(
            polymer["molecule_type"] in STANDARD_ALPHABETS
            and set(polymer["sequence"])
            <= STANDARD_ALPHABETS[polymer["molecule_type"]]
            for polymer in record["polymers"]
        )
    ]
    records = sorted(records, key=lambda row: (estimated_tokens(row), row["pdb_id"]))
    tracks = sorted({track for record in records for track in record["tracks"]})
    selected: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    for track in tracks:
        for record in records:
            if track not in record["tracks"] or counts[track] >= per_track:
                continue
            selected[record["pdb_id"]] = record
            counts[track] += 1
            if counts[track] == per_track:
                break

    missing = {track: per_track - counts[track] for track in tracks if counts[track] < per_track}
    if missing:
        raise ValueError(f"not enough FoldBench inputs for requested coverage: {missing}")
    return sorted(selected.values(), key=lambda row: row["pdb_id"])


def to_protenix(record: dict[str, Any]) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    for polymer in sorted(record["polymers"], key=lambda row: int(row["entity_id"])):
        molecule_type = polymer["molecule_type"]
        if molecule_type not in POLYMER_KEYS:
            raise ValueError(f"unsupported polymer type {molecule_type!r}")
        sequences.append(
            {
                POLYMER_KEYS[molecule_type]: {
                    "sequence": polymer["sequence"],
                    "count": assembly_count(
                        record, polymer["entity_id"], polymer["asym_ids"]
                    ),
                    "modifications": [],
                }
            }
        )

    for nonpolymer in sorted(
        record["nonpolymers"], key=lambda row: int(row["entity_id"])
    ):
        sequences.append(
            {
                "ligand": {
                    "ligand": f"CCD_{nonpolymer['ccd_code']}",
                    "count": assembly_count(
                        record, nonpolymer["entity_id"], nonpolymer["asym_ids"]
                    ),
                }
            }
        )

    return {"name": record["pdb_id"], "sequences": sequences}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--per-track",
        type=int,
        default=0,
        help="Deterministically sample this many assemblies per track; 0 keeps all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    if args.per_track < 0:
        raise ValueError("--per-track must be non-negative")
    selected = select_per_track(records, args.per_track) if args.per_track else records
    converted = [to_protenix(record) for record in selected]

    target_counts = Counter(
        row["track"] for record in selected for row in record["target_rows"]
    )
    manifest = {
        "schema": "protenix-official-foldbench-input-v1",
        "contains_ground_truth_coordinates": False,
        "source": str(args.input_jsonl),
        "selection": {
            "per_track": args.per_track,
            "assemblies": len(selected),
            "targets_by_track": dict(sorted(target_counts.items())),
            "pdb_ids": [record["pdb_id"] for record in selected],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

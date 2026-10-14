#!/usr/bin/env python3
"""Build compact, group-safe train/validation indexes from FoldingTrain-100K."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ESMC_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--validation-permille", type=int, default=20)
    parser.add_argument("--seed", default="foldingtrain-reference-v1")
    parser.add_argument("--smoke-train", type=int, default=2048)
    parser.add_argument("--smoke-validation", type=int, default=256)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_int(seed: str, value: str) -> int:
    return int(hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(), 16)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def balanced_subset(
    records: list[dict[str, Any]], count: int, seed: str
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    tasks = sorted(by_task)
    selected = []
    remaining = count
    for index, task in enumerate(tasks):
        quota = remaining // (len(tasks) - index)
        ranked = sorted(
            by_task[task],
            key=lambda record: stable_int(seed + task, record["example_id"]),
        )
        chosen = ranked[: min(quota, len(ranked))]
        selected.extend(chosen)
        remaining -= len(chosen)
    if remaining:
        chosen_ids = {record["example_id"] for record in selected}
        remainder = sorted(
            (record for record in records if record["example_id"] not in chosen_ids),
            key=lambda record: stable_int(seed + "remainder", record["example_id"]),
        )
        selected.extend(remainder[:remaining])
    return sorted(selected, key=lambda record: record["example_id"])


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_permille < 1000:
        raise ValueError("--validation-permille must be between 1 and 999")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rich_by_structure: dict[str, dict[str, Any]] = {}
    compact = []
    for record in read_jsonl(args.manifest):
        structure_path = record["structure_path"]
        if "sequences" in record:
            rich_by_structure[structure_path] = record
        rich = record if "sequences" in record else rich_by_structure.get(structure_path)
        if rich is None:
            raise ValueError(f"no base metadata for {record['example_id']}")

        sequences = []
        for sequence in rich.get("sequences", []):
            digest = sequence["sha256"]
            feature_path = (
                args.data_root / "features" / ESMC_VERSION / f"{digest}.safetensors"
            )
            sequences.append(
                {
                    "chain_id": sequence["chain_id"],
                    "molecule_type": sequence["molecule_type"],
                    "length": int(sequence["length"]),
                    "sequence": sequence["sequence"],
                    "sequence_sha256": digest,
                    "esmc_path": str(feature_path),
                    "esmc_available": feature_path.is_file()
                    and feature_path.stat().st_size > 0,
                }
            )
        structure_group = hashlib.sha256(structure_path.encode()).hexdigest()
        compact.append(
            {
                "example_id": record["example_id"],
                "task": record["task"],
                "source": record["source"],
                "source_id": record["source_id"],
                "selection_view": record.get("selection_view", "task_view"),
                "structure_path": structure_path,
                "structure_group": structure_group,
                "token_count": int(
                    rich.get("token_count")
                    or record.get("token_count")
                    or sum(int(value["length"]) for value in sequences)
                ),
                "sequences": sequences,
            }
        )

    groups = {
        record["structure_group"]:
        stable_int(args.seed + "validation", record["structure_group"]) % 1000
        < args.validation_permille
        for record in compact
    }
    train = [record for record in compact if not groups[record["structure_group"]]]
    validation = [record for record in compact if groups[record["structure_group"]]]
    smoke_train = balanced_subset(train, args.smoke_train, args.seed + "smoke-train")
    smoke_validation = balanced_subset(
        validation, args.smoke_validation, args.seed + "smoke-validation"
    )
    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
        "smoke_train": args.output_dir / "smoke_train.jsonl",
        "smoke_validation": args.output_dir / "smoke_validation.jsonl",
    }
    for name, records in (
        ("train", train),
        ("validation", validation),
        ("smoke_train", smoke_train),
        ("smoke_validation", smoke_validation),
    ):
        write_jsonl(paths[name], records)

    audit = {
        "schema_version": "folding-reference-index-v1",
        "source_manifest": str(args.manifest),
        "seed": args.seed,
        "validation_permille": args.validation_permille,
        "examples": len(compact),
        "unique_structure_groups": len(groups),
        "splits": {
            name: {
                "examples": len(records),
                "unique_structure_groups": len(
                    {record["structure_group"] for record in records}
                ),
                "task_counts": dict(sorted(Counter(record["task"] for record in records).items())),
                "missing_esmc_sequence_refs": sum(
                    not sequence["esmc_available"]
                    for record in records
                    for sequence in record["sequences"]
                    if sequence["molecule_type"] == "protein"
                ),
            }
            for name, records in (
                ("train", train),
                ("validation", validation),
                ("smoke_train", smoke_train),
                ("smoke_validation", smoke_validation),
            )
        },
        "group_overlap": len(
            {record["structure_group"] for record in train}
            & {record["structure_group"] for record in validation}
        ),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (args.output_dir / "index.audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

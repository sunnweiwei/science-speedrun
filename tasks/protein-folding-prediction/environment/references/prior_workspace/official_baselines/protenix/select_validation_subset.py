#!/usr/bin/env python3
"""Select a deterministic, track-covered common-validation health check."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from agent_pipeline.contract import FOLDBENCH_TRACKS, load_json, sha256_file


QUOTAS = {
    track: (4 if track == "interface_antibody_antigen" else 3)
    for track in FOLDBENCH_TRACKS
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _select(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task = row.get("unified_task", "")
        if task in QUOTAS:
            grouped[task].append(row)
    selected: list[dict[str, str]] = []
    used_pdbs: set[str] = set()
    for task in FOLDBENCH_TRACKS:
        candidates = sorted(
            grouped[task],
            key=lambda row: hashlib.sha256(
                f"{seed}:{task}:{row['unified_example_id']}".encode()
            ).digest(),
        )
        task_rows = []
        for row in candidates:
            pdb_id = row["pdb_id"].lower()
            if pdb_id in used_pdbs:
                continue
            task_rows.append(row)
            used_pdbs.add(pdb_id)
            if len(task_rows) == QUOTAS[task]:
                break
        if len(task_rows) != QUOTAS[task]:
            raise RuntimeError(
                f"common validation has only {len(task_rows)} unique rows for {task}"
            )
        selected.extend(task_rows)
    return sorted(selected, key=lambda row: (row["pdb_id"], row["unified_example_id"]))


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite validation subset: {args.output_dir}")
    audit = load_json(args.input_audit)
    if audit["native_manifest_sha256"] != sha256_file(args.input_index):
        raise ValueError("validation index does not match its native-view audit")
    with gzip.open(args.input_index, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not rows:
        raise ValueError("validation index is empty")
    selected = _select(rows, args.seed)
    args.output_dir.mkdir(parents=True)
    index = args.output_dir / "valid.csv.gz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    index.write_bytes(buffer.getvalue())
    pdb_ids = args.output_dir / "pdb_ids.txt"
    pdb_ids.write_text("".join(f"{row['pdb_id'].lower()}\n" for row in selected))
    counts = Counter(row["unified_task"] for row in selected)
    report = {
        "schema_version": "foldingtrain-common-validation-healthcheck/v1",
        "source_audit": str(args.input_audit),
        "source_index_sha256": sha256_file(args.input_index),
        "seed": args.seed,
        "examples": len(selected),
        "unique_pdbs": len({row["pdb_id"].lower() for row in selected}),
        "task_counts": dict(sorted(counts.items())),
        "index": str(index),
        "index_sha256": sha256_file(index),
        "pdb_ids": str(pdb_ids),
        "selection_features": ["unified_task", "unified_example_id"],
        "foldbench_ground_truth_used": False,
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

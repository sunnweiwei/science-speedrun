#!/usr/bin/env python3
"""Select a deterministic, task-balanced PDB subset for upstream data prep."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--seed", default="protenix-official-smoke-v1")
    parser.add_argument("--required-split", default="train")
    return parser.parse_args()


def stable_rank(seed: str, record: dict[str, Any]) -> str:
    value = f"{seed}\0{record['task']}\0{record['pdb_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    args = arguments()
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with args.manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("unified_split") != args.required_split
                or record.get("source") != "openfold3_pdb"
                or int(record.get("token_count", 10**9)) > args.max_tokens
            ):
                continue
            by_task[record["task"]].setdefault(record["pdb_id"].lower(), record)

    selected = []
    for task in sorted(by_task):
        ranked = sorted(
            by_task[task].values(), key=lambda record: stable_rank(args.seed, record)
        )
        selected.extend(ranked[: args.per_task])
    selected.sort(key=lambda record: (record["pdb_id"], record["task"]))
    unique_by_pdb = {
        record["pdb_id"].lower(): record for record in selected
    }
    unique = [unique_by_pdb[pdb_id] for pdb_id in sorted(unique_by_pdb)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pdb_ids.txt").write_text(
        "".join(f"{record['pdb_id'].lower()}\n" for record in unique)
    )
    (args.output_dir / "rcsb_mmcif_urls.txt").write_text(
        "".join(
            f"https://files.rcsb.org/download/{record['pdb_id'].upper()}.cif.gz\n"
            for record in unique
        )
    )
    audit = {
        "schema_version": "protenix-official-smoke-selection-v1",
        "source_manifest": str(args.manifest),
        "seed": args.seed,
        "required_split": args.required_split,
        "per_task": args.per_task,
        "max_tokens": args.max_tokens,
        "task_selections": len(selected),
        "structures": len(unique),
        "task_counts": dict(sorted(Counter(item["task"] for item in selected).items())),
        "pdb_ids": [item["pdb_id"].lower() for item in unique],
    }
    (args.output_dir / "selection.audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

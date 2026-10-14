#!/usr/bin/env python3
"""Merge deterministic structure-token shards into the local DPLM data view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--staging-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    from datasets import Dataset

    args.output.mkdir(parents=True, exist_ok=True)
    audits = {}
    for split in ("train", "valid"):
        rows = []
        shard_audits = []
        for rank in range(args.world_size):
            shard = args.shards / f"{split}.rank-{rank:03d}.jsonl"
            audit_path = shard.with_suffix(shard.suffix + ".audit.json")
            if not shard.is_file() or not audit_path.is_file():
                raise FileNotFoundError(f"missing shard or audit for {split} rank {rank}")
            shard_audit = json.loads(audit_path.read_text())
            if not shard_audit["quality_comparable"]:
                raise ValueError(f"limited smoke shard cannot be finalized: {shard}")
            shard_audits.append(shard_audit)
            rows.extend(json.loads(line) for line in shard.open())
        rows.sort(key=lambda row: row["pdb_name"])
        if len(rows) != len({row["pdb_name"] for row in rows}):
            raise AssertionError(f"duplicate example IDs in {split} shards")
        Dataset.from_list(rows).save_to_disk(str(args.output / split))
        audits[split] = {
            "records": len(rows),
            "missing_backbone_residues": sum(
                audit["missing_backbone_residues"] for audit in shard_audits
            ),
            "shards": [
                {"rank": rank, "sha256": sha256(args.shards / f"{split}.rank-{rank:03d}.jsonl")}
                for rank in range(args.world_size)
            ],
        }
    source = json.loads(args.staging_audit.read_text())
    report = {
        "schema_version": "foldingtrain-dplm2-native-view-v1",
        "source_adapter_audit": source,
        "splits": audits,
        "data_policy": "supported projection of frozen unified split; no upstream dataset substitution",
    }
    (args.output / "adapter_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

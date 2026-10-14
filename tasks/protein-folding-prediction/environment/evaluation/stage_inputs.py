#!/usr/bin/env python3
"""Stage a coordinate-free FoldBench request subset for direct model replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("/workspace/data/foldbench/suite"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/workspace/data/foldbench/features.esmc6b_final_int8.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output must be empty: {args.output}")

    manifest = json.loads((args.suite / "suite_manifest.json").read_text())
    wanted = set(args.request_id)
    entries = [
        entry for entry in manifest["requests"]
        if not wanted or entry["request_id"] in wanted
    ]
    if wanted != {entry["request_id"] for entry in entries}:
        raise KeyError(f"unknown request IDs: {sorted(wanted - {entry['request_id'] for entry in entries})}")
    if not wanted:
        entries = entries[: args.limit]

    requests = []
    required = set()
    for entry in entries:
        path = args.suite / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"request checksum mismatch: {entry['request_id']}")
        request = json.loads(path.read_text())
        requests.append(request)
        for entity in request["target"]["entities"]:
            if entity["type"] == "protein":
                required.add(hashlib.sha256(entity["sequence"].encode()).hexdigest())

    feature_rows = {}
    for line in args.features.read_text().splitlines():
        if line:
            row = json.loads(line)
            feature_rows[row["sequence_sha256"]] = row
    missing = required - feature_rows.keys()
    if missing:
        raise ValueError(f"missing feature rows: {sorted(missing)[:3]}")

    feature_dir = args.output / "features/esmc6b-final-int8"
    feature_dir.mkdir(parents=True)
    public_rows = []
    for digest in sorted(required):
        row = dict(feature_rows[digest])
        source = (args.features.parent / row["path"]).resolve()
        relative = Path("features/esmc6b-final-int8") / f"{digest}.safetensors"
        destination = args.output / relative
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        row["path"] = relative.as_posix()
        public_rows.append(row)
    (args.output / "requests.jsonl").write_text(
        "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests)
    )
    (args.output / "features.esmc6b_final_int8.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows)
    )
    print(json.dumps({
        "requests": [request["request_id"] for request in requests],
        "unique_protein_features": len(public_rows),
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

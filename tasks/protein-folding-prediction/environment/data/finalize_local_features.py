#!/usr/bin/env python3
"""Finalize organizer-generated shared features into the releasable data layout."""

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


FEATURE_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"
FEATURE_WIDTH = 2560
GROUP_SIZE = 64


def atomic_jsonl(path, rows):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def link_or_copy(source, destination):
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace conflicting feature: {destination}")
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(source, destination)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: finalize_local_features.py DATA_ROOT")
    root = Path(sys.argv[1]).resolve()
    shared = root / f"features/{FEATURE_VERSION}"

    generation_manifest = root / "manifests/foldingtrain-100k-v1/features.esmc6b_mixed256.jsonl"
    training_manifest = root / "manifests/foldingtrain-100k-v1/features.esmc6b_final_int8.jsonl"
    training_rows = []
    with generation_manifest.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            feature = shared / f"{row['sequence_sha256']}.safetensors"
            if not feature.is_file() or feature.stat().st_size == 0:
                raise FileNotFoundError(feature)
            training_rows.append({
                "sequence_sha256": row["sequence_sha256"],
                "sequence": row["sequence"],
                "length": row["length"],
                "shape": [row["length"], FEATURE_WIDTH],
                "dtype": "int8",
                "scale_shape": [row["length"], FEATURE_WIDTH // GROUP_SIZE],
                "scale_dtype": "bfloat16",
                "quantization": "symmetric-int8",
                "group_size": GROUP_SIZE,
                "feature_version": FEATURE_VERSION,
                "local_status": "ready",
                "local_bytes": feature.stat().st_size,
                "local_path": f"/data/features/{FEATURE_VERSION}/{feature.name}",
            })

    generation = root / "foldbench/features.esmc6b_mixed256.generation.jsonl"
    suite_store = root / f"foldbench/features/{FEATURE_VERSION}"
    suite_store.mkdir(parents=True, exist_ok=True)
    suite_rows = []
    with generation.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            sequence = record["sequence"]
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            if digest != record["sequence_sha256"] or len(sequence) != record["length"]:
                raise ValueError(f"Invalid generation record: {digest}")
            source = shared / f"{digest}.safetensors"
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            destination = suite_store / source.name
            link_or_copy(source, destination)
            suite_rows.append({
                "sequence_sha256": digest,
                "length": len(sequence),
                "shape": [len(sequence), FEATURE_WIDTH],
                "dtype": "int8",
                "scale_shape": [len(sequence), FEATURE_WIDTH // GROUP_SIZE],
                "scale_dtype": "bfloat16",
                "quantization": "symmetric-int8",
                "group_size": GROUP_SIZE,
                "feature_version": FEATURE_VERSION,
                "path": f"features/{FEATURE_VERSION}/{source.name}",
            })

    atomic_jsonl(training_manifest, training_rows)
    atomic_jsonl(root / "foldbench/features.esmc6b_final_int8.jsonl", suite_rows)
    print(json.dumps({
        "training_features": len(training_rows),
        "foldbench_features": len(suite_rows),
        "shared_store": str(shared),
        "foldbench_store": str(suite_store),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

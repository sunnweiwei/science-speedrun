#!/usr/bin/env python3
"""Validate the complete materialized FoldBench data release without model code."""

import hashlib
import json
from pathlib import Path
import re
import struct
import sys


FEATURE_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"
FEATURE_WIDTH = 2560
GROUP_SIZE = 64
EXPECTED_TRAINING_FEATURES = 91_366
EXPECTED_SUITE_FEATURES = 1_568
ESMFOLD2_FAST_SHA256 = "60ca19f2898188beba92944365f7b909efd9c99212f5018af75cc47cd9a6184a"
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
FEATURE_BOUNDARY = "ESMCModel.transformer.final_layer_norm"
KERNEL_BACKEND = "pytorch-sdpa-no-xformers-no-transformer-engine"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(root, value):
    value = Path(value)
    if value.is_absolute():
        try:
            value = value.relative_to("/data")
        except ValueError as error:
            raise ValueError(f"Feature path is outside /data: {value}") from error
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or escaping feature file: {value}")
    return path


def validate_feature_payload(path, digest, length):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > min(16 * 1024 * 1024, file_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid safetensors header: {path}") from error
    metadata = header.pop("__metadata__", {}) if isinstance(header, dict) else {}
    if not isinstance(header, dict) or set(header) != {"values", "scales"}:
        raise ValueError(f"Invalid canonical feature tensors: {digest}")
    expected = {
        "values": ("I8", [length, FEATURE_WIDTH], length * FEATURE_WIDTH),
        "scales": (
            "BF16", [length, FEATURE_WIDTH // GROUP_SIZE],
            length * (FEATURE_WIDTH // GROUP_SIZE) * 2,
        ),
    }
    segments = []
    for name, (dtype, shape, byte_count) in expected.items():
        item = header.get(name)
        offsets = item.get("data_offsets") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or item.get("dtype") != dtype
                or item.get("shape") != shape or not isinstance(offsets, list)
                or len(offsets) != 2 or not all(isinstance(value, int) for value in offsets)
                or offsets[1] - offsets[0] != byte_count):
            raise ValueError(f"Invalid canonical feature tensor {name}: {digest}")
        segments.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(segments):
        if start != cursor or end < start:
            raise ValueError(f"Noncontiguous canonical feature payload: {digest}")
        cursor = end
    if cursor != file_size - 8 - header_size:
        raise ValueError(f"Canonical feature payload length mismatch: {digest}")
    required_metadata = {
        "sequence_sha256": digest,
        "esmc_repo": "biohub/ESMC-6B",
        "esmc_revision": ESMC_REVISION,
        "boundary": FEATURE_BOUNDARY,
        "kernel_backend": KERNEL_BACKEND,
        "schema_version": FEATURE_VERSION,
        "quantization": "symmetric-int8",
        "group_size": str(GROUP_SIZE),
        "scale_dtype": "bfloat16",
    }
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in required_metadata.items()
    ):
        raise ValueError(f"Invalid canonical feature provenance: {digest}")


def read_jsonl(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def validate_feature_manifest(path, root, expected):
    records = {}
    for line_number, record in read_jsonl(path):
        digest = record.get("sequence_sha256")
        length = record.get("length")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid digest at {path}:{line_number}")
        if digest in records:
            raise ValueError(f"Duplicate feature digest: {digest}")
        if (record.get("local_status", "ready") != "ready"
                or record.get("feature_version") != FEATURE_VERSION
                or record.get("dtype") != "int8"
                or record.get("scale_dtype") != "bfloat16"
                or record.get("quantization") != "symmetric-int8"
                or record.get("group_size") != GROUP_SIZE
                or record.get("shape") != [length, FEATURE_WIDTH]
                or record.get("scale_shape") != [length, FEATURE_WIDTH // GROUP_SIZE]):
            raise ValueError(f"Incomplete feature record: {digest}")
        if "sequence" in record:
            sequence = record["sequence"]
            if len(sequence) != length or hashlib.sha256(sequence.encode()).hexdigest() != digest:
                raise ValueError(f"Sequence metadata mismatch: {digest}")
        feature = contained_file(root, record.get("path", record.get("local_path", "")))
        validate_feature_payload(feature, digest, length)
        records[digest] = length
    if len(records) != expected:
        raise ValueError(f"{path} contains {len(records)} features; expected {expected}")
    return records


def suite_proteins(root):
    suite = root / "foldbench/suite"
    manifest = json.loads((suite / "suite_manifest.json").read_text())
    required = {}
    for item in manifest["requests"]:
        request = json.loads(contained_file(suite, item["path"]).read_text())
        for entity in request["target"]["entities"]:
            if entity["type"] == "protein":
                sequence = entity["sequence"]
                required[hashlib.sha256(sequence.encode()).hexdigest()] = len(sequence)
    return required


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_release.py MATERIALIZED_DATA_ROOT")
    root = Path(sys.argv[1]).resolve()
    training = validate_feature_manifest(
        root / "manifests/foldingtrain-100k-v1/features.esmc6b_final_int8.jsonl",
        root,
        EXPECTED_TRAINING_FEATURES,
    )
    suite = validate_feature_manifest(
        root / "foldbench/features.esmc6b_final_int8.jsonl",
        root / "foldbench",
        EXPECTED_SUITE_FEATURES,
    )
    required = suite_proteins(root)
    if suite != required:
        raise ValueError("FoldBench feature manifest does not exactly cover suite proteins")
    baseline = root / "models/esmfold2-fast-b28d8ace/model.safetensors"
    if sha256(baseline) != ESMFOLD2_FAST_SHA256:
        raise ValueError("ESMFold2-Fast baseline checkpoint checksum mismatch")
    if any("esmc-6b" in path.name.lower() for path in (root / "models").rglob("*")):
        raise ValueError("The Agent data release must not contain an ESMC-6B checkpoint")
    print(json.dumps({
        "training_features": len(training),
        "foldbench_features": len(suite),
        "esmc_checkpoint_exposed": False,
        "esmfold2_fast_reference_sha256": ESMFOLD2_FAST_SHA256,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

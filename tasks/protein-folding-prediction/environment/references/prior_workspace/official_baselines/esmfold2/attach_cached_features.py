#!/usr/bin/env python3
"""Attach validated frozen ESMC mixed-256 tensors to an ESMFold2 native view."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from agent_pipeline.contract import load_json, sha256_file
from data_pipeline.cache_esmc_mixed256 import FEATURE_VERSION


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _data_path(path: str, data_root: Path) -> Path:
    value = PurePosixPath(path)
    if value.is_absolute():
        try:
            value = value.relative_to("/data")
        except ValueError as error:
            raise ValueError(f"feature path is outside /data: {path}") from error
    if ".." in value.parts:
        raise ValueError(f"feature path escapes data root: {path}")
    return data_root.joinpath(*value.parts)


def _uri(path: Path, data_root: Path) -> str:
    relative = path.resolve().relative_to(data_root.resolve())
    return f"data://{relative.as_posix()}"


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite native view: {args.output_dir}")
    source_audit = load_json(args.input_audit)
    if source_audit["native_manifest_sha256"] != sha256_file(args.input_manifest):
        raise ValueError("input native manifest does not match its audit")

    features: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(args.feature_manifest):
        if row.get("local_status") != "ready":
            continue
        if row.get("feature_version") != FEATURE_VERSION:
            raise ValueError("feature manifest mixes incompatible feature versions")
        sequence = row["sequence"]
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if digest != row["sequence_sha256"]:
            raise ValueError("feature manifest sequence digest mismatch")
        if row.get("shape") != [len(sequence), 256]:
            raise ValueError("feature manifest shape mismatch")
        path = _data_path(row["local_path"], args.data_root)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        features[digest] = {**row, "resolved_path": path}

    accepted = []
    rejected = Counter(source_audit["rejection_reasons"])
    for record in _read_jsonl(args.input_manifest):
        # v2 views retain the complete assembly.  ESMC is defined only on
        # protein chains; RNA/DNA/ligand tokens receive the same zero LM signal
        # produced by Biohub's online mixed-biomolecule path.
        if "input" in record:
            protein_inputs = [
                item
                for item in record["input"]["sequences"]
                if item["type"] == "protein"
            ]
            attached: dict[str, dict[str, str]] = {}
            missing = False
            for item in protein_inputs:
                sequence = item["sequence"]
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                feature = features.get(digest)
                if feature is None:
                    missing = True
                    break
                if feature["sequence"] != sequence:
                    raise ValueError(f"feature sequence collision: {record['example_id']}")
                attached[item["id"]] = {
                    "sequence_sha256": digest,
                    "uri": _uri(feature["resolved_path"], args.data_root),
                }
            if missing:
                rejected["missing_esmc_mixed256"] += 1
                continue
            accepted.append(
                {
                    **record,
                    "esmc_mixed256": attached,
                    "esmc_feature_version": FEATURE_VERSION,
                }
            )
            continue

        # Read compatibility for the already-audited v1 monomer projection.
        digest = hashlib.sha256(record["sequence"].encode()).hexdigest()
        if digest != record["sequence_sha256"]:
            raise ValueError(f"native sequence digest mismatch: {record['example_id']}")
        feature = features.get(digest)
        if feature is None:
            rejected["missing_esmc_mixed256"] += 1
            continue
        if feature["sequence"] != record["sequence"]:
            raise ValueError(f"feature sequence collision: {record['example_id']}")
        accepted.append({
            **record,
            "esmc_mixed256": _uri(feature["resolved_path"], args.data_root),
            "esmc_feature_version": FEATURE_VERSION,
        })

    args.output_dir.mkdir(parents=True)
    output_manifest = args.output_dir / f"{args.split}.jsonl"
    output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted)
    )
    audit = {
        **source_audit,
        "accepted_examples": len(accepted),
        "rejected_examples": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "native_manifest": str(output_manifest),
        "native_manifest_sha256": sha256_file(output_manifest),
    }
    if audit["accepted_examples"] + audit["rejected_examples"] != audit["source_examples"]:
        raise RuntimeError("cached native view does not partition the source split")
    (args.output_dir / "native_view_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    details = {
        "feature_version": FEATURE_VERSION,
        "feature_manifest": str(args.feature_manifest),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "precomputed_boundary": "base_z_linear + base_z_combine; before base_z_mlp",
        "accepted_examples": len(accepted),
    }
    (args.output_dir / "native_view_details.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**audit, "details": details}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

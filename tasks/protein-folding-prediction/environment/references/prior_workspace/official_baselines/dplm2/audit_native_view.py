#!/usr/bin/env python3
"""Bind the finalized DPLM-2 capability projection to the DatasetPackage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_pipeline.contract import (
    NATIVE_VIEW_AUDIT_SCHEMA,
    load_json,
    sha256_file,
    validate_dataset_package,
    validate_native_view_audit,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--native-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    package = load_json(args.dataset_package)
    validate_dataset_package(package, args.data_root)
    source = load_json(args.native_view / "adapter_audit.json")
    train = source["source_adapter_audit"]["counts"]["train"]
    accepted = int(source["splits"]["train"]["records"])
    source_examples = int(train["global_examples"])
    reasons = {
        "unsupported_track": int(train["folding_unsupported_task"]),
        "not_single_protein": int(train["folding_not_single_protein"]),
        "sequence_too_long": int(train["folding_too_long"]),
    }
    if accepted + sum(reasons.values()) != source_examples:
        raise AssertionError("DPLM-2 adapter reasons do not partition the train split")
    native_manifest = args.native_view / "train" / "data-00000-of-00001.arrow"
    if not native_manifest.is_file():
        raise FileNotFoundError(native_manifest)
    audit = {
        "schema_version": NATIVE_VIEW_AUDIT_SCHEMA,
        "model_id": "dplm2_50m",
        "dataset_id": package["dataset_id"],
        "source_split_sha256": package["splits"]["train"]["sha256"],
        "source_examples": source_examples,
        "accepted_examples": accepted,
        "rejected_examples": sum(reasons.values()),
        "rejection_reasons": reasons,
        "native_manifest": str(native_manifest.resolve()),
        "native_manifest_sha256": sha256_file(native_manifest),
        "selection_policy": "capability_only",
    }
    validate_native_view_audit(audit, package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed OpenDDE native-view preflight using the released featurizer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from agent_pipeline.contract import sha256_file
from official_baselines.opendde.data_adapter import iter_manifest, load_example


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--log-interval", type=int, default=1000)
    return parser.parse_args()


def classify(error: Exception) -> str:
    message = str(error).lower()
    if "ccd" in message or "component" in message or "reference info" in message:
        return "released_featurizer_missing_ccd"
    if "target atoms mapped" in message:
        return "insufficient_aligned_target_atoms"
    return "released_featurizer_error"


def main() -> None:
    args = arguments()
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    from opendde.data.core import ccd

    components = args.ccd_dir / "components.cif"
    rdkit = args.ccd_dir / "components.cif.rdkit_mol.pkl"
    for path in (components, rdkit):
        if not path.is_file():
            raise FileNotFoundError(path)
    ccd.set_ccd_cache_paths(str(components), str(rdkit))
    source_audit = json.loads(args.input_audit.read_text())
    if source_audit["native_manifest_sha256"] != sha256_file(args.input_manifest):
        raise ValueError("input manifest does not match its audit")
    accepted = []
    rejected: Counter[str] = Counter(source_audit["rejection_reasons"])
    task_counts: Counter[str] = Counter()
    processed = 0
    for record in iter_manifest(args.input_manifest):
        processed += 1
        try:
            features, target, target_mask = load_example(record, args.data_root)
            if int(features["token_index"].shape[0]) > int(record["token_count"]):
                raise ValueError("released tokenization exceeds declared source token count")
            del features, target, target_mask
        except Exception as error:
            rejected[classify(error)] += 1
            continue
        accepted.append(record)
        task_counts[record["task"]] += 1
        if processed % args.log_interval == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "accepted": len(accepted),
                        "runtime_rejected": processed - len(accepted),
                    }
                ),
                flush=True,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.split}.jsonl"
    manifest.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in accepted))
    audit = {
        **source_audit,
        "accepted_examples": len(accepted),
        "rejected_examples": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "native_manifest": str(manifest),
        "native_manifest_sha256": sha256_file(manifest),
    }
    if audit["accepted_examples"] + audit["rejected_examples"] != audit["source_examples"]:
        raise RuntimeError("final native audit does not partition the source split")
    (args.output_dir / "native_view_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    details = {
        "preflight": "released OpenDDE SampleDictToFeatures + exact target alignment",
        "preflight_examples": processed,
        "accepted_task_counts": dict(sorted(task_counts.items())),
        "ccd_components_sha256": sha256_file(components),
        "ccd_rdkit_cache_sha256": sha256_file(rdkit),
    }
    (args.output_dir / "native_view_details.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**audit, "details": details}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

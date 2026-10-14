#!/usr/bin/env python3
"""Create a self-contained DPLM-2 ModelBundle from a completed campaign run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from agent_pipeline.contract import (
    MODEL_BUNDLE_SCHEMA,
    load_json,
    make_tree_portable,
    sha256_file,
    sha256_tree,
    validate_model_bundle,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--struct-tokenizer", type=Path, required=True)
    parser.add_argument("--native-view-audit", type=Path, required=True)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image", default="protein-folding:dplm2-runtime-v6")
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument(
        "--bundle-id",
        default="dplm2-50m-foldingtrain100k-pass1-seed20260802-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ModelBundle: {args.output}")
    for path in (args.model_dir, args.struct_tokenizer):
        if not path.is_dir():
            raise FileNotFoundError(path)
    package = load_json(args.dataset_package)
    native_audit = load_json(args.native_view_audit)
    if native_audit["model_id"] != "dplm2_50m":
        raise ValueError("native view audit belongs to a different model")
    export_audit = load_json(args.model_dir / "export_manifest.json")
    if export_audit["deployment_parameters"] != 47_053_446:
        raise ValueError("unexpected DPLM-2 deployment parameter count")

    payload = args.output / "payload"
    shutil.copytree(args.model_dir, payload / "model")
    shutil.copytree(args.struct_tokenizer, payload / "struct_tokenizer")
    shutil.copy2(args.native_view_audit, args.output / "native_view_audit.json")
    inference_config = {
        "schema_version": "dplm2-contract-inference/v1",
        "max_iter": 500,
        "temperature": 1.0,
        "unmasking_strategy": "stochastic1.0",
        "sampling_strategy": "annealing@2.0:0.1",
        "model_path": "payload/model",
        "struct_tokenizer_path": "payload/struct_tokenizer",
    }
    inference_path = args.output / "inference_config.json"
    inference_path.write_text(
        json.dumps(inference_config, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "bundle_id": args.bundle_id,
        "model_id": "dplm2_50m",
        "dataset": {
            "dataset_id": package["dataset_id"],
            "train_sha256": package["splits"]["train"]["sha256"],
            "native_view_audit": "native_view_audit.json",
        },
        "training": {
            # Exact DDP exposure includes deterministic repeated batches used
            # to equalize rank lengths; the unique native projection is 42,912.
            "examples_seen": 47_781,
            "optimizer_steps": 369,
            "epochs": 1.0,
            "seed": 20260802,
            "completed": True,
        },
        "artifacts": {
            "checkpoint": {
                "path": "payload",
                "sha256": sha256_tree(payload),
                "kind": "tree",
            },
            "model_config": {
                "path": "payload/model/config.json",
                "sha256": sha256_file(payload / "model" / "config.json"),
            },
            "inference_config": {
                "path": "inference_config.json",
                "sha256": sha256_file(inference_path),
            },
        },
        "inference": {
            "runtime_image": args.runtime_image,
            "runtime_digest": args.runtime_digest,
            "command": [
                "python",
                "-m",
                "official_baselines.dplm2.contract_inference",
                "--request",
                "{request}",
                "--model-dir",
                "/bundle/payload/model",
                "--struct-tokenizer",
                "/bundle/payload/struct_tokenizer",
                "--output-dir",
                "{output_dir}",
                "--result",
                "{result}",
                "--max-iter",
                "500",
            ],
            "request_schema": "foldbench-folding-request/v0",
            "result_schema": "foldbench-folding-result/v0",
        },
        "capabilities": {
            "eval_tracks": ["monomer_protein"],
            "entity_types": ["protein", "rna", "dna", "ligand"],
            "msa": False,
            "templates": False,
        },
        "provenance": {
            "upstream_revision": "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d",
            "deployment_parameters": 47_053_446,
            "unique_native_examples": 42_912,
            "ddp_repeated_example_exposures": 4_869,
            "padded_token_slots": 9_860_962,
            "sequence_pretraining_steps": {"mlm": 735, "dplm": 735},
            "folding_recipe": "released dplm2_150m non-data recipe",
            "input_projection": "unique protein entity; ignore non-protein context and symmetry copies",
            "unsupported_tracks_reported_as": "N/S",
        },
    }
    manifest_path = args.output / "model_bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    make_tree_portable(args.output)
    validate_model_bundle(manifest, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

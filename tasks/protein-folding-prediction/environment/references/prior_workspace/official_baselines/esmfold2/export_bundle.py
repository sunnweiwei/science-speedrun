#!/usr/bin/env python3
"""Export a completed scaled ESMFold2 run as a portable ModelBundle."""

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
from official_baselines.esmfold2.config import (
    ESMC_REVISION,
    ESM_REVISION,
    MODEL_NAME_50M,
    TRANSFORMERS_REVISION,
)
from official_baselines.esmfold2.trainable_forward import persistent_state_dict


HF_RUNTIME_CACHE = "/opt/hf/hub"
CCD_RUNTIME_CACHE = (
    "/opt/hf/hub/models--biohub--ESMFold2/snapshots/"
    "1ebf0e3481a5184eb6171d40615c79e384b48796"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-view-audit", type=Path, required=True)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image", default="protein-folding:esmc6b-runtime-v1")
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--examples-seen", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _ignore_source(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {".git", "__pycache__", ".pytest_cache", "wandb"}
        or name.endswith(".pyc")
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ModelBundle: {args.output}")
    for path in (args.checkpoint, args.native_view_audit, args.dataset_package):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.examples_seen < 1 or args.optimizer_steps < 1:
        raise ValueError("training counts must be positive")
    package = load_json(args.dataset_package)
    native_audit = load_json(args.native_view_audit)
    if native_audit.get("model_id") != "esmfold2_fast_50m":
        raise ValueError("native view audit belongs to a different model")

    import torch

    training_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in training_checkpoint or "ema" not in training_checkpoint:
        raise ValueError("ESMFold2 training checkpoint is missing model/EMA weights")
    deployment_state = persistent_state_dict(training_checkpoint["model"])
    unknown_ema = set(training_checkpoint["ema"]) - set(deployment_state)
    if unknown_ema:
        raise ValueError(f"EMA has unknown parameter keys: {sorted(unknown_ema)[:3]}")
    deployment_state.update(persistent_state_dict(training_checkpoint["ema"]))

    payload = args.output / "payload"
    checkpoint_dir = payload / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    deployment_checkpoint = checkpoint_dir / f"{MODEL_NAME_50M}.pt"
    torch.save(
        {
            "model": deployment_state,
            "step": training_checkpoint.get("step", args.optimizer_steps),
        },
        deployment_checkpoint,
    )
    project = payload / "runtime" / "project"
    shutil.copytree(
        args.project_root / "official_baselines",
        project / "official_baselines",
        ignore=_ignore_source,
    )
    for name in ("folding_contract.py", "folding_adapters.py"):
        shutil.copy2(args.project_root / name, project / name)
    shutil.copy2(args.native_view_audit, args.output / "native_view_audit.json")

    model_config = {
        "model_name": MODEL_NAME_50M,
        "deployment_parameters": 49_198_301,
        "transformers_revision": TRANSFORMERS_REVISION,
        "esm_revision": ESM_REVISION,
        "esmc_revision": ESMC_REVISION,
        "folding_trunk_layers": 8,
        "parcae_coda": True,
    }
    model_config_path = payload / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=2, sort_keys=True) + "\n")
    inference_config = {
        "schema_version": "esmfold2-contract-inference/v1",
        "loops": 3,
        "sampling_steps": 20,
        "num_samples": "from request",
        "max_eval_tokens": 2048,
    }
    inference_path = args.output / "inference_config.json"
    inference_path.write_text(json.dumps(inference_config, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "bundle_id": args.bundle_id,
        "model_id": "esmfold2_fast_50m",
        "dataset": {
            "dataset_id": package["dataset_id"],
            "train_sha256": package["splits"]["train"]["sha256"],
            "native_view_audit": "native_view_audit.json",
        },
        "training": {
            "examples_seen": args.examples_seen,
            "optimizer_steps": args.optimizer_steps,
            "epochs": args.examples_seen / native_audit["accepted_examples"],
            "seed": args.seed,
            "completed": True,
        },
        "artifacts": {
            "checkpoint": {"path": "payload", "sha256": sha256_tree(payload), "kind": "tree"},
            "model_config": {"path": "payload/model_config.json", "sha256": sha256_file(model_config_path)},
            "inference_config": {"path": "inference_config.json", "sha256": sha256_file(inference_path)},
        },
        "inference": {
            "runtime_image": args.runtime_image,
            "runtime_digest": args.runtime_digest,
            "command": [
                "env",
                "PYTHONPATH=/bundle/payload/runtime/project:/opt/transformers/src:/opt/esm",
                "HF_HOME=/opt/hf",
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
                "python",
                "-m",
                "official_baselines.esmfold2.contract_inference",
                "--request", "{request}",
                "--checkpoint", f"/bundle/payload/checkpoint/{MODEL_NAME_50M}.pt",
                "--hf-home", HF_RUNTIME_CACHE,
                "--ccd-dir", CCD_RUNTIME_CACHE,
                "--output-dir", "{output_dir}",
                "--result", "{result}",
                "--loops", "3",
                "--sampling-steps", "20",
            ],
            "request_schema": "foldbench-folding-request/v0",
            "result_schema": "foldbench-folding-result/v0",
        },
        "capabilities": {
            "eval_tracks": [
                "monomer_protein", "monomer_rna", "monomer_dna",
                "interface_protein_protein", "interface_antibody_antigen",
                "interface_protein_peptide", "interface_protein_rna",
                "interface_protein_dna", "interface_protein_ligand",
            ],
            "entity_types": ["protein", "rna", "dna", "ligand"],
            "msa": False,
            "templates": False,
            "max_eval_tokens": 2048,
        },
        "provenance": {
            "transformers_revision": TRANSFORMERS_REVISION,
            "esm_revision": ESM_REVISION,
            "esmc_revision": ESMC_REVISION,
            "deployment_parameters": 49_198_301,
            "frozen_esmc_parameters_in_budget": False,
            "recipe": "foldingtrain_edm_v1 project-owned reconstruction",
            "checkpoint_kind": "EMA",
            "confidence_head_training": "frozen",
            "unsupported_requests_reported_as": "N/S only for per-request token limits",
            "input_projection": "full Biohub StructurePredictionInput biomolecular complex",
        },
    }
    manifest_path = args.output / "model_bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    make_tree_portable(args.output)
    validate_model_bundle(manifest, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

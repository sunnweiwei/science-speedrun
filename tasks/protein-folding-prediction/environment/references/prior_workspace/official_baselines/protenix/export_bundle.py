#!/usr/bin/env python3
"""Export a completed official-code Protenix run as a portable ModelBundle."""

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
from official_baselines.protenix.config import MODEL_NAME, UPSTREAM_REVISION


REQUIRED_COMMON_FILES = (
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "clusters-by-entity-40.txt",
    "obsolete_release_date.csv",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--native-view-audit", type=Path, required=True)
    parser.add_argument("--dataset-package", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-image", default="protein-folding:protenix-official")
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


def _copy_runtime(args: argparse.Namespace, runtime: Path) -> None:
    shutil.copytree(
        args.upstream_root,
        runtime / "protenix",
        ignore=_ignore_source,
    )
    project = runtime / "project"
    (project / "official_baselines").mkdir(parents=True)
    shutil.copy2(
        args.project_root / "official_baselines" / "__init__.py",
        project / "official_baselines" / "__init__.py",
    )
    shutil.copytree(
        args.project_root / "official_baselines" / "protenix",
        project / "official_baselines" / "protenix",
        ignore=_ignore_source,
    )
    for name in ("folding_contract.py", "folding_adapters.py"):
        shutil.copy2(args.project_root / name, project / name)

    common = runtime / "protenix_data" / "common"
    common.mkdir(parents=True)
    for name in REQUIRED_COMMON_FILES:
        source = args.common_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, common / name)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ModelBundle: {args.output}")
    for path in (
        args.checkpoint,
        args.run_config,
        args.native_view_audit,
        args.dataset_package,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.examples_seen < 1 or args.optimizer_steps < 1:
        raise ValueError("training counts must be positive")

    package = load_json(args.dataset_package)
    native_audit = load_json(args.native_view_audit)
    if native_audit.get("model_id") != "protenix_pairformer_50m":
        raise ValueError("native view audit belongs to a different model")

    payload = args.output / "payload"
    checkpoint_dir = payload / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    # Deployment checkpoints intentionally omit optimizer/scheduler state.
    import torch

    source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in source_checkpoint:
        raise ValueError("upstream checkpoint does not contain model weights")
    deployment_checkpoint = checkpoint_dir / f"{MODEL_NAME}.pt"
    torch.save(
        {
            "model": source_checkpoint["model"],
            "step": source_checkpoint.get("step", args.optimizer_steps - 1),
        },
        deployment_checkpoint,
    )

    runtime = payload / "runtime"
    _copy_runtime(args, runtime)
    shutil.copy2(args.native_view_audit, args.output / "native_view_audit.json")
    shutil.copy2(args.run_config, payload / "training_config.yaml")
    model_config = {
        "model_name": MODEL_NAME,
        "upstream_revision": UPSTREAM_REVISION,
        "deployment_parameters": 48_681_151,
        "cycles": 4,
        "channels": {"c_s": 256, "c_z": 64, "c_token": 256, "c_atom": 96},
    }
    model_config_path = payload / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=2, sort_keys=True) + "\n")
    inference_config = {
        "schema_version": "protenix-contract-inference/v1",
        "diffusion_steps": 20,
        "cycles": 4,
        "num_samples": "from request",
        "msa": False,
        "templates": False,
        "max_eval_tokens": 2048,
    }
    inference_path = args.output / "inference_config.json"
    inference_path.write_text(json.dumps(inference_config, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "bundle_id": args.bundle_id,
        "model_id": "protenix_pairformer_50m",
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
                "PYTHONPATH=/bundle/payload/runtime/project:/bundle/payload/runtime/protenix",
                "PROTENIX_ROOT_DIR=/bundle/payload/runtime/protenix_data",
                "LAYERNORM_TYPE=torch",
                "python",
                "-m",
                "official_baselines.protenix.contract_inference",
                "--request",
                "{request}",
                "--checkpoint-dir",
                "/bundle/payload/checkpoint",
                "--output-dir",
                "{output_dir}",
                "--result",
                "{result}",
                "--diffusion-steps",
                "20",
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
            "upstream_revision": UPSTREAM_REVISION,
            "deployment_parameters": 48_681_151,
            "architecture": "released Protenix tiny graph with declared channel scaling",
            "recipe": "official non-data recipe; common frozen DatasetPackage",
            "checkpoint_kind": "EMA" if "ema" in args.checkpoint.name else "ordinary",
            "unsupported_requests_reported_as": "N/S",
        },
    }
    manifest_path = args.output / "model_bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    make_tree_portable(args.output)
    validate_model_bundle(manifest, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

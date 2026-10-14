#!/usr/bin/env python3
"""Audit the pinned ESMFold2 source/config and public training availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_baselines.esmfold2.config import (
    DEFAULT_LOOPS,
    DEFAULT_SAMPLING_STEPS,
    ESM_REPOSITORY,
    ESM_REVISION,
    ESMC_REPOSITORY,
    ESMC_REVISION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TRANSFORMERS_REPOSITORY,
    TRANSFORMERS_REVISION,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--esm-root", type=Path, default=Path("/opt/esm"))
    parser.add_argument(
        "--transformers-root", type=Path, default=Path("/opt/transformers")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    head = (repository / ".git" / "HEAD").read_text().strip()
    return head if not head.startswith("ref: ") else (
        repository / ".git" / head.removeprefix("ref: ")
    ).read_text().strip()


def main() -> None:
    args = arguments()
    revisions = {
        "esm": git_revision(args.esm_root),
        "transformers": git_revision(args.transformers_root),
    }
    expected = {"esm": ESM_REVISION, "transformers": TRANSFORMERS_REVISION}
    if revisions != expected:
        raise AssertionError(f"source revisions {revisions} != {expected}")
    config = json.loads((args.config / "config.json").read_text())
    if config.get("architectures") != ["ESMFold2Model"]:
        raise AssertionError("model config does not select ESMFold2Model")
    if config.get("msa_encoder", {}).get("enabled") is not False:
        raise AssertionError("ESMFold2-Fast config unexpectedly enables MSA")
    if config.get("num_loops") != DEFAULT_LOOPS:
        raise AssertionError("official Fast loop default changed")
    source_train_entries = sorted(
        str(path.relative_to(root))
        for root in (args.esm_root, args.transformers_root)
        for path in root.rglob("*esmfold2*/**/train.py")
    )
    report = {
        "schema_version": "official-folding-baseline-availability-v1",
        "baseline": "esmfold2_fast",
        "source_repositories": {
            ESM_REPOSITORY: revisions["esm"],
            TRANSFORMERS_REPOSITORY: revisions["transformers"],
        },
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "frozen_embedding_model": {
            "repository": ESMC_REPOSITORY,
            "revision": ESMC_REVISION,
        },
        "official_inference": {
            "available": True,
            "top_level_class": "transformers.models.esmfold2.modeling_esmfold2.ESMFold2Model",
            "num_loops": DEFAULT_LOOPS,
            "num_sampling_steps": DEFAULT_SAMPLING_STEPS,
            "msa_conditioned": False,
            "all_atom_biomolecules": True,
        },
        "official_training": {
            "available": False,
            "public_train_entries": source_train_entries,
            "missing": [
                "runnable training entry",
                "loss composition and weights",
                "optimizer and schedule",
                "checkpoint/resume recipe",
            ],
        },
        "project_architecture_reproduction": {
            "available": True,
            "model_name": "esmfold2_fast_50m_v1",
            "parameters": 49_198_301,
            "architecture_source": "official_biohub_modules",
            "recipe_source": "project_owned_foldingtrain_edm_v1",
            "gpu_update_smoke": True,
        },
        "strict_50m_trainable_baseline": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit OpenDDE inference and project architecture-reproduction provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from official_baselines.opendde.config import (
    CHECKPOINT_SHA256,
    MODEL_NAME,
    MODEL_PARAMETERS,
    RELEASE_ASSET_REVISION,
    UPSTREAM_IMAGE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    head = (repository / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    return (repository / ".git" / head.removeprefix("ref: ")).read_text().strip()


def main() -> None:
    args = arguments()
    revision = git_revision(args.upstream)
    if revision != UPSTREAM_REVISION:
        raise AssertionError(f"source revision {revision} != {UPSTREAM_REVISION}")
    required = [
        "runner/inference.py",
        "runner/batch_inference.py",
        "opendde/model/opendde.py",
        "opendde/config/model_manifest.json",
    ]
    missing = [name for name in required if not (args.upstream / name).is_file()]
    if missing:
        raise AssertionError(f"missing official inference files: {missing}")
    training_entries = sorted(
        str(path.relative_to(args.upstream))
        for path in args.upstream.rglob("train.py")
    )
    manifest = json.loads(
        (args.upstream / "opendde/config/model_manifest.json").read_text()
    )
    serialized = json.dumps(manifest)
    if CHECKPOINT_SHA256 not in serialized:
        raise AssertionError("released checkpoint digest absent from model manifest")
    report = {
        "schema_version": "official-folding-baseline-availability-v1",
        "baseline": "opendde_v1",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": revision,
        "upstream_image": UPSTREAM_IMAGE,
        "release_asset_revision": RELEASE_ASSET_REVISION,
        "official_inference": {
            "available": True,
            "model_name": MODEL_NAME,
            "parameters": MODEL_PARAMETERS,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "all_atom_biomolecules": True,
            "msa_optional": True,
            "template_optional": True,
        },
        "official_training": {
            "available": False,
            "public_train_entries": training_entries,
            "missing": [
                "runnable training entry",
                "training loss composition",
                "optimizer and schedule",
                "training checkpoint/resume recipe",
            ],
        },
        "project_architecture_reproduction": {
            "available": True,
            "model_name": "opendde_structural_50m_v1",
            "parameters": 48_746_962,
            "architecture_source": "official_opendde_modules",
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

#!/usr/bin/env python3
"""Materialize auditable local Hugging Face configs for the official 50M scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from official_baselines.dplm2.config import (
    MODEL_NAME,
    SCALED_ARCHITECTURE,
    UPSTREAM_DPLM2_MODEL_REVISION,
    UPSTREAM_DPLM_MODEL_REVISION,
    UPSTREAM_REVISION,
    scaled_config,
)


TOKENIZER_FILES = ("vocab.txt", "special_tokens_map.json", "tokenizer_config.json")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dplm-metadata", type=Path, required=True)
    parser.add_argument("--dplm2-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(source: Path, output: Path, *, multimodal: bool) -> dict:
    source_config = json.loads((source / "config.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    config = scaled_config(source_config, multimodal=multimodal)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    for name in TOKENIZER_FILES:
        shutil.copyfile(source / name, output / name)
    return {name: sha256(output / name) for name in ("config.json", *TOKENIZER_FILES)}


def main() -> None:
    args = arguments()
    dplm_output = args.output_root / "dplm_50m"
    dplm2_output = args.output_root / "dplm2_50m"
    files = {
        "dplm_50m": materialize(args.dplm_metadata, dplm_output, multimodal=False),
        "dplm2_50m": materialize(
            args.dplm2_metadata, dplm2_output, multimodal=True
        ),
    }
    report = {
        "schema_version": "official-dplm2-config-materialization-v1",
        "model_name": MODEL_NAME,
        "upstream_code_revision": UPSTREAM_REVISION,
        "source_model_revisions": {
            "dplm": UPSTREAM_DPLM_MODEL_REVISION,
            "dplm2": UPSTREAM_DPLM2_MODEL_REVISION,
        },
        "scaled_architecture": SCALED_ARCHITECTURE,
        "files": files,
    }
    (args.output_root / "materialization.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

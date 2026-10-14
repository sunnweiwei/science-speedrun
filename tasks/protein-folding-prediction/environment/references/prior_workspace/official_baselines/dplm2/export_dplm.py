#!/usr/bin/env python3
"""Export an official DPLM Lightning checkpoint as a local HF initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

    model = DiffusionProteinLanguageModel.from_pretrained(
        str(args.checkpoint), from_huggingface=False
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.net.save_pretrained(args.output, safe_serialization=True)
    model.tokenizer.save_pretrained(args.output)
    manifest = {
        "schema_version": "official-dplm-hf-export-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "top_level_class": f"{type(model.net).__module__}.{type(model.net).__name__}",
        "parameters": sum(parameter.numel() for parameter in model.net.parameters()),
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a trained upstream DPLM-2 checkpoint as a merged local HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    # The saved Hydra config references this upstream environment variable.
    os.environ.setdefault("PROJECT_ROOT", os.environ.get("DPLM_ROOT", "/opt/dplm"))

    from peft.peft_model import PeftModel
    from byprot.models.dplm2.dplm2 import (
        MultimodalDiffusionProteinLanguageModel,
    )

    model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
        str(args.checkpoint), from_huggingface=False
    )
    used_peft = isinstance(model.net, PeftModel)
    if used_peft:
        model.net = model.net.merge_and_unload()
    # resize_token_embeddings() starts from the sequence-DPLM config and the
    # released training path never updates this dispatch metadata. Inference
    # dispatches solely on dplm_type, so bind the already-instantiated official
    # EsmForDPLM2 weights to their actual registered class.
    if type(model.net).__name__ != "EsmForDPLM2":
        raise TypeError(f"unexpected merged DPLM-2 network: {type(model.net)}")
    model.net.config.dplm_type = "dplm2_esm"
    args.output.mkdir(parents=True, exist_ok=True)
    model.net.save_pretrained(args.output, safe_serialization=True)
    model.tokenizer.save_pretrained(args.output)
    weights = args.output / "model.safetensors"
    report = {
        "schema_version": "official-dplm2-hf-export-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "peft_merged": used_peft,
        "dplm_type": model.net.config.dplm_type,
        "deployment_parameters": sum(
            parameter.numel() for parameter in model.net.parameters()
        ),
        "top_level_class": f"{type(model.net).__module__}.{type(model.net).__name__}",
        "weights_sha256": sha256(weights),
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit provenance, topology, and parameter count of the official-code model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from official_baselines.protenix.config import (
    CHANNEL_OVERRIDES,
    EXPECTED_PARAMETER_RANGE,
    MODEL_NAME,
    UPSTREAM_IMAGE,
    UPSTREAM_MODEL_NAME,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    deep_update,
    register_model,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/official"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def count(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def read_git_revision(repository: Path) -> str:
    """Read HEAD without changing container-global Git safe-directory config."""
    git_dir = repository / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text().strip()
    for line in (git_dir / "packed-refs").read_text().splitlines():
        if line and not line.startswith(("#", "^")):
            revision, name = line.split(" ", 1)
            if name == ref:
                return revision
    raise RuntimeError(f"cannot resolve {ref} in {repository}")


def main() -> None:
    args = arguments()
    register_model()

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    from protenix.model.protenix import Protenix

    base = deepcopy({**configs_base, "data": data_configs})
    deep_update(base, deepcopy(model_configs[MODEL_NAME]))
    base["triangle_attention"] = "torch"
    base["triangle_multiplicative"] = "torch"
    config = parse_configs(base, fill_required_with_null=True)
    model = Protenix(config)
    parameters = count(model)
    low, high = EXPECTED_PARAMETER_RANGE
    if not low <= parameters <= high:
        raise AssertionError(
            f"parameter count {parameters:,} is outside [{low:,}, {high:,}]"
        )
    if type(model).__module__ != "protenix.model.protenix":
        raise AssertionError(f"unexpected top-level class: {type(model)}")

    revision = read_git_revision(args.upstream)
    if revision != UPSTREAM_REVISION:
        raise AssertionError(
            f"upstream revision {revision} does not match pin {UPSTREAM_REVISION}"
        )

    trainable_origins = sorted(
        {
            type(module).__module__
            for module in model.modules()
            if any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        }
    )
    foreign_origins = [
        origin
        for origin in trainable_origins
        if not (origin.startswith("protenix.") or origin.startswith("torch.nn."))
    ]
    if foreign_origins:
        raise AssertionError(f"non-upstream trainable modules: {foreign_origins}")

    canonical_overrides = json.dumps(
        CHANNEL_OVERRIDES, sort_keys=True, separators=(",", ":")
    ).encode()
    report = {
        "schema_version": "official-folding-baseline-audit-v1",
        "model_name": MODEL_NAME,
        "upstream_model_name": UPSTREAM_MODEL_NAME,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": revision,
        "upstream_image": UPSTREAM_IMAGE,
        "top_level_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": parameters,
        "channel_overrides": CHANNEL_OVERRIDES,
        "channel_overrides_sha256": hashlib.sha256(canonical_overrides).hexdigest(),
        "topology": {
            "recycles": int(config.model.N_cycle),
            "msa_blocks": len(model.msa_module.blocks),
            "pairformer_blocks": len(model.pairformer_stack.blocks),
            "diffusion_atom_encoder_blocks": int(
                model.diffusion_module.atom_attention_encoder.atom_transformer.n_blocks
            ),
            "diffusion_transformer_blocks": int(
                model.diffusion_module.diffusion_transformer.n_blocks
            ),
            "diffusion_atom_decoder_blocks": int(
                model.diffusion_module.atom_attention_decoder.atom_transformer.n_blocks
            ),
            "confidence_blocks": int(model.confidence_head.n_blocks),
        },
        "component_parameters": {
            "input_embedder": count(model.input_embedder),
            "msa_module": count(model.msa_module),
            "pairformer": count(model.pairformer_stack),
            "diffusion": count(model.diffusion_module),
            "confidence": count(model.confidence_head),
            "distogram": count(model.distogram_head),
        },
        "trainable_module_origins": trainable_origins,
        "foreign_trainable_module_origins": foreign_origins,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the hybrid uses only the official ESM input path and folding modules."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from first_party_baselines.hybrid_esmc_pairformer.config import (
    ESMC_MODEL,
    ESMC_REVISION,
    EXPECTED_PARAMETER_RANGE,
    FEATURE_ID,
    HYBRID_OVERRIDES,
    MODEL_NAME,
    register_model,
)
from official_baselines.protenix.config import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    deep_update,
)
from official_baselines.protenix.verify import read_git_revision


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/official"))
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


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
        raise AssertionError(f"parameters {parameters:,} outside [{low:,}, {high:,}]")
    if not model.input_embedder.esm_configs["enable"]:
        raise AssertionError("official Protenix ESM input path is disabled")
    projection = model.input_embedder.linear_esm
    if projection.in_features != 256 or projection.out_features != config.c_s_inputs:
        raise AssertionError("unexpected ESMC projection dimensions")
    revision = read_git_revision(args.upstream)
    if revision != UPSTREAM_REVISION:
        raise AssertionError(f"source revision {revision} != {UPSTREAM_REVISION}")
    origins = sorted(
        {
            type(module).__module__
            for module in model.modules()
            if any(parameter.requires_grad for parameter in module.parameters(recurse=False))
        }
    )
    foreign = [
        origin
        for origin in origins
        if not (origin.startswith("protenix.") or origin.startswith("torch.nn."))
    ]
    if foreign:
        raise AssertionError(f"unexpected trainable module origins: {foreign}")
    report = {
        "schema_version": "first-party-folding-baseline-audit-v1",
        "model_name": MODEL_NAME,
        "claim": "first-party composition using unmodified official Protenix modules",
        "folding_source": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": revision,
        },
        "frozen_embedding_source": {
            "repository": ESMC_MODEL,
            "revision": ESMC_REVISION,
            "feature_id": FEATURE_ID,
            "dimension": 256,
        },
        "parameters_excluding_frozen_esmc": parameters,
        "hybrid_overrides": HYBRID_OVERRIDES,
        "esm_projection_parameters": count(projection),
        "trainable_module_origins": origins,
        "foreign_trainable_module_origins": foreign,
        "non_data_recipe": "official Protenix trainer/loss/optimizer/scheduler/checkpoint path",
        "data": "frozen unified FoldingTrain split with cached ESMC features",
    }
    if args.feature_manifest:
        from first_party_baselines.hybrid_esmc_pairformer.data_adapter import (
            CachedESMCFeatures,
        )

        report["data_adapter"] = CachedESMCFeatures(args.feature_manifest).audit()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

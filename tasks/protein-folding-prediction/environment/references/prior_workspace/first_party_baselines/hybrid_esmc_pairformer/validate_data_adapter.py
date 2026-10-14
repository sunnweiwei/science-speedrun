#!/usr/bin/env python3
"""Exercise cached ESMC injection on a real common-data Protenix batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from first_party_baselines.hybrid_esmc_pairformer.data_adapter import (
    install_cached_feature_adapter,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--unified-train-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    import torch
    import yaml
    from ml_collections import ConfigDict

    cache = install_cached_feature_adapter(args.feature_manifest)
    from protenix.data.pipeline.dataloader import get_dataloaders

    config = ConfigDict(yaml.safe_load(args.training_config.read_text()))
    config.data.num_dl_workers = 0
    config.data.epoch_size = 1
    config.data.test_sets = []
    for name in config.data.train_sets:
        config.data[name].base_info.random_sample_if_failed = False
    train_loader, _ = get_dataloaders(config, world_size=1, seed=config.seed)
    batch = next(iter(train_loader))
    features = batch["input_feature_dict"]
    if "esm_token_embedding" not in features:
        raise AssertionError("adapter did not attach esm_token_embedding")
    embedding = features["esm_token_embedding"]
    if embedding.ndim != 2 or embedding.shape[-1] != 256:
        raise AssertionError(f"unexpected ESMC feature shape: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all() or not torch.count_nonzero(embedding):
        raise AssertionError("real Protenix batch has empty or non-finite ESMC features")

    pdb_id = str(batch["basic"]["pdb_id"]).lower()
    common_ids = {
        str(json.loads(line).get("pdb_id", "")).lower()
        for line in args.unified_train_split.open()
    }
    if pdb_id not in common_ids:
        raise AssertionError(f"native batch {pdb_id} is outside the frozen train split")
    report = {
        "schema_version": "hybrid-esmc-protenix-data-adapter-smoke-v1",
        "quality_comparable": False,
        "pdb_id": pdb_id,
        "pdb_id_in_frozen_train_split": True,
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "embedding_nonzero": int(torch.count_nonzero(embedding)),
        "cache": cache.audit(),
        "official_dataset_class": (
            "protenix.data.pipeline.dataset.BaseSingleDataset"
        ),
        "official_crop_and_feature_pipeline_retained": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

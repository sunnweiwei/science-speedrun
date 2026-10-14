#!/usr/bin/env python3
"""Launch the pinned upstream DPLM-2 trainer with the official recipe locked."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import runpy
import sys

from official_baselines.dplm2.data_adapter import (
    install as install_data_adapter,
    install_subprocess_bootstrap,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--dplm-config", type=Path, required=True)
    parser.add_argument("--dplm2-config", type=Path, required=True)
    parser.add_argument("--dplm-initialization", type=Path, required=True)
    parser.add_argument("--dataset-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="dplm2_official_50m_v1")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="complete passes over the supported common-data projection",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--steps-per-pass",
        type=int,
        help="audited DDP step count for one supported-data pass (required in full mode)",
    )
    parser.add_argument("--upstream", type=Path, default=Path("/opt/dplm"))
    return parser.parse_args()


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def main() -> None:
    args = arguments()
    if args.passes < 1:
        raise ValueError("--passes must be >= 1")
    require(args.upstream / "train.py", "pinned upstream trainer")
    require(args.dplm_config / "config.json", "50M DPLM config")
    require(args.dplm2_config / "config.json", "50M DPLM-2 config")
    require(args.dplm_initialization / "config.json", "50M DPLM initialization")
    require(args.dataset_view / "train", "unified DPLM-2 training view")
    require(args.dataset_view / "valid", "unified DPLM-2 validation view")
    require(args.dataset_view / "adapter_audit.json", "unified-data adapter audit")
    args.output_root.mkdir(parents=True, exist_ok=True)

    # The released dplm2_150m YAML still emits model.vocab_size, but the
    # DPLM2Config dataclass at the same pinned revision has removed that field.
    # The live implementation derives the combined vocabulary from tokenizer,
    # so delete only this stale YAML key and fail if the API changes again.
    from byprot.models.dplm2.dplm2 import DPLM2Config

    declared_fields = {field.name for field in fields(DPLM2Config)}
    stale_release_keys = {"vocab_size", "freeze_backbone"}
    revived = stale_release_keys & declared_fields
    if revived:
        raise RuntimeError(
            f"pinned DPLM2Config now consumes {sorted(revived)}; "
            "remove the corresponding compatibility override"
        )

    # Architecture, objectives, optimizer and checkpoint behavior resolve from
    # the official experiment.  Full campaign mode additionally fixes data
    # exposure to --passes so every candidate sees a controlled budget.
    overrides = [
        "experiment=dplm2/dplm2_150m",
        f"name={args.run_name}",
        f"paths.data_dir={args.dataset_view.parent}",
        f"paths.log_dir={args.output_root / args.run_name}",
        f"datamodule.csv_file={args.dataset_view.name}",
        f"datamodule.vocab_file={args.dplm2_config}",
        f"model.tokenizer.vocab_file={args.dplm2_config}",
        f"model.net.name={args.dplm_config}",
        f"model.net.pretrained_model_name_or_path={args.dplm_initialization}",
        f"trainer.devices={args.devices}",
        f"seed={args.seed}",
        f"train.seed={args.seed}",
        "~model.vocab_size",
        "~model.freeze_backbone",
    ]
    if args.profile == "smoke":
        # Explicitly non-comparable: exercise data -> loss -> optimizer -> ckpt.
        overrides.extend(
            [
                "trainer.max_steps=1",
                "trainer.num_sanity_val_steps=0",
                "trainer.val_check_interval=1",
                "+trainer.limit_val_batches=1",
                "datamodule.max_tokens=256",
                "datamodule.max_len=128",
                "datamodule.num_workers=0",
            ]
        )
    else:
        if args.steps_per_pass is None or args.steps_per_pass < 1:
            raise ValueError("full mode requires --steps-per-pass >= 1")
        overrides.extend(
            [
                f"trainer.min_epochs={args.passes}",
                f"trainer.max_epochs={args.passes}",
                f"trainer.max_steps={args.steps_per_pass * args.passes}",
                "trainer.check_val_every_n_epoch=1",
                "trainer.val_check_interval=1.0",
            ]
        )
    install_subprocess_bootstrap()
    install_data_adapter()
    sys.argv = [str(args.upstream / "train.py"), *overrides]
    runpy.run_path(str(args.upstream / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

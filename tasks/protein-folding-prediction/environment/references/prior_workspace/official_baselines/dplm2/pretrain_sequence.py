#!/usr/bin/env python3
"""Run the two pinned official sequence stages needed by scaled DPLM-2."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from official_baselines.dplm2.data_adapter import (
    install as install_data_adapter,
    install_subprocess_bootstrap,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mlm", "dplm"), required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mlm-checkpoint", type=Path)
    parser.add_argument("--sequence-dataset-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
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
    require(args.config / "config.json", "scaled sequence config")
    require(args.sequence_dataset_view / "train", "unified sequence training view")
    require(args.sequence_dataset_view / "valid", "unified sequence validation view")
    require(
        args.sequence_dataset_view / "adapter_audit.json",
        "unified sequence-data adapter audit",
    )
    if args.stage == "dplm":
        if args.mlm_checkpoint is None:
            raise ValueError("--mlm-checkpoint is required for official DPLM stage 2")
        require(args.mlm_checkpoint, "MLM stage-1 checkpoint")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.stage}_official_50m_v1"

    experiment = "dplm/mlm_150m" if args.stage == "mlm" else "dplm/dplm_150m_stage2"
    overrides = [
        f"experiment={experiment}",
        f"name={run_name}",
        f"paths.data_dir={args.sequence_dataset_view.parent}",
        f"paths.log_dir={args.output_root / run_name}",
        f"datamodule.data_dir={args.sequence_dataset_view}",
        f"model.net.name={args.config}",
        f"trainer.devices={args.devices}",
        f"seed={args.seed}",
        f"train.seed={args.seed}",
    ]
    if args.stage == "dplm":
        overrides.append(
            f"model.net.pretrained_model_name_or_path={args.mlm_checkpoint}"
        )
    else:
        # The released MLM config omits the field required by its own current
        # model factory.  False is the intended stage-1 behavior: initialize
        # the scaled topology locally instead of fetching ESM2-150M.
        overrides.append("+model.net.pretrain=false")
    if args.profile == "smoke":
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
        # Paper defaults target a much larger dataset.  The controlled
        # campaign instead fixes exposure to complete common-data passes.
        overrides.extend(
            [
                f"trainer.min_epochs={args.passes}",
                f"trainer.max_epochs={args.passes}",
                f"trainer.max_steps={args.steps_per_pass * args.passes}",
                "trainer.check_val_every_n_epoch=1",
                "trainer.val_check_interval=1.0",
            ]
        )
    install_subprocess_bootstrap(args.config)
    install_data_adapter(args.config)
    sys.argv = [str(args.upstream / "train.py"), *overrides]
    runpy.run_path(str(args.upstream / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

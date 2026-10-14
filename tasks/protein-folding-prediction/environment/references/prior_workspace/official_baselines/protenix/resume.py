"""Resume the upstream Protenix trainer without replaying sampled examples.

Upstream checkpoints restore the optimizer, scheduler, and step counter, but
the upstream distributed sampler starts again at the beginning of its seeded
epoch. The campaign is a fixed exposure budget, so a resumed process must
continue at the corresponding sampler offset instead of replaying the prefix.

This module only slices the deterministic index stream. Model code, feature
construction, losses, optimizer, scheduler, and checkpoint loading all remain
the pinned upstream implementations.
"""

from __future__ import annotations

from itertools import islice
import os
from pathlib import Path
import sys
from typing import Iterator


ENV_SKIP_STEPS = "AUTOFOLD_RESUME_SKIP_OPTIMIZER_STEPS"
ENV_CHECKPOINT = "AUTOFOLD_RESUME_CHECKPOINT"
ENV_EMA_CHECKPOINT = "AUTOFOLD_RESUME_EMA_CHECKPOINT"


def _option(name: str, argv: list[str]) -> str | None:
    """Read one upstream ``--key value`` or ``--key=value`` option."""
    result = None
    for index, value in enumerate(argv):
        if value == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            result = argv[index + 1]
        elif value.startswith(f"{name}="):
            result = value.split("=", 1)[1]
    return result


def resume_settings(
    environ: dict[str, str] | None = None,
    argv: list[str] | None = None,
) -> tuple[Path, int] | None:
    """Validate and return the checkpoint and sampler offset for a resume."""
    env = os.environ if environ is None else environ
    values = sys.argv if argv is None else argv
    checkpoint_value = env.get(ENV_CHECKPOINT, "")
    ema_checkpoint_value = env.get(ENV_EMA_CHECKPOINT, "")
    skip_value = env.get(ENV_SKIP_STEPS, "")
    if not checkpoint_value and not ema_checkpoint_value and not skip_value:
        return None
    if not checkpoint_value or not ema_checkpoint_value or not skip_value:
        raise ValueError(
            f"{ENV_CHECKPOINT}, {ENV_EMA_CHECKPOINT}, and {ENV_SKIP_STEPS} "
            "must be set together"
        )

    checkpoint = Path(checkpoint_value)
    ema_checkpoint = Path(ema_checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not ema_checkpoint.is_file():
        raise FileNotFoundError(ema_checkpoint)
    try:
        skip_steps = int(skip_value)
    except ValueError as error:
        raise ValueError(f"{ENV_SKIP_STEPS} must be an integer") from error
    if skip_steps <= 0:
        raise ValueError(f"{ENV_SKIP_STEPS} must be positive")

    checkpoint_step = checkpoint.stem
    if not checkpoint_step.isdigit() or int(checkpoint_step) + 1 != skip_steps:
        raise ValueError(
            "resume checkpoint name must be <step>.pt and agree with the sampler offset"
        )
    if _option("--load_checkpoint_path", values) != str(checkpoint):
        raise ValueError("upstream --load_checkpoint_path differs from the resume checkpoint")
    if _option("--load_ema_checkpoint_path", values) != str(ema_checkpoint):
        raise ValueError("upstream --load_ema_checkpoint_path differs from the EMA checkpoint")
    if (_option("--load_params_only", values) or "true").lower() != "false":
        raise ValueError("resume must restore optimizer, scheduler, and step state")
    if (_option("--load_strict", values) or "false").lower() != "true":
        raise ValueError("resume must strictly load every model state key")
    if (_option("--skip_load_step", values) or "false").lower() != "false":
        raise ValueError("resume must restore the upstream step counter")
    return checkpoint, skip_steps


def install(environ: dict[str, str] | None = None, argv: list[str] | None = None) -> int:
    """Install a fail-closed offset on the upstream distributed sampler."""
    settings = resume_settings(environ=environ, argv=argv)
    if settings is None:
        return 0
    _, skip_steps = settings

    from protenix.data.pipeline.dataloader import DistributedWeightedSampler

    original_iter = DistributedWeightedSampler.__iter__

    def resumed_iter(sampler: object) -> Iterator[int]:
        return islice(original_iter(sampler), skip_steps, None)

    DistributedWeightedSampler.__iter__ = resumed_iter
    return skip_steps

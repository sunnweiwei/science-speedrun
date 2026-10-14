"""Transparent project-owned training utilities for inference-only releases.

Nothing in this module is claimed to be an upstream training Recipe.  It is a
small, shared implementation of the EDM objective needed to make released
all-atom diffusion architectures trainable on FoldingTrain-100K.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


RECIPE_NAME = "foldingtrain_edm_v1"
RECIPE_PROVENANCE = "project_owned"


@dataclass(frozen=True)
class ProjectRecipeConfig:
    """The fully explicit non-data Recipe used for unpublished trainers."""

    sigma_data: float = 16.0
    noise_log_mean: float = -1.2
    noise_log_std: float = 1.5
    diffusion_weight: float = 1.0
    distogram_weight: float = 0.03
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 1_000
    total_steps: int = 100_000
    gradient_clip_norm: float = 1.0

    def metadata(self) -> dict[str, Any]:
        return {
            "name": RECIPE_NAME,
            "provenance": RECIPE_PROVENANCE,
            "official_recipe": False,
            "values": asdict(self),
        }


def sample_noise_levels(batch_size: int, config: ProjectRecipeConfig, device):
    """Sample the log-normal noise distribution published in ESMFold2 config."""

    import torch

    normal = torch.randn(batch_size, device=device, dtype=torch.float32)
    return config.sigma_data * torch.exp(
        config.noise_log_mean + config.noise_log_std * normal
    )


def edm_coordinate_loss(
    predicted,
    target,
    sigma,
    atom_mask,
    *,
    sigma_data: float,
):
    """Masked, EDM-preconditioned coordinate denoising loss.

    ``predicted`` and ``target`` end in ``[N_atom, 3]``.  ``sigma`` contains
    one value for each leading sample/batch element and ``atom_mask`` ends in
    ``[N_atom]``.  Missing coordinates never contribute to the denominator.
    """

    sigma = sigma.float()
    weight = (sigma.square() + sigma_data**2) / (
        (sigma * sigma_data).square().clamp_min(1.0e-12)
    )
    while weight.ndim < predicted.ndim:
        weight = weight.unsqueeze(-1)
    mask = atom_mask.to(dtype=predicted.dtype).unsqueeze(-1)
    while mask.ndim < predicted.ndim:
        mask = mask.unsqueeze(-3)
    squared_error = (predicted.float() - target.float()).square()
    numerator = (weight * squared_error * mask).sum()
    denominator = (mask.sum() * predicted.shape[-1]).clamp_min(1.0)
    return numerator / denominator


def add_coordinate_noise(target, sigma):
    """Create ``x_noisy`` without detaching the denoising computation graph."""

    import torch

    scale = sigma
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return target + scale * torch.randn_like(target)


def cosine_learning_rate(step: int, config: ProjectRecipeConfig) -> float:
    """Linear warmup followed by cosine decay to ten percent of peak LR."""

    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, config.total_steps - config.warmup_steps - 1
    )
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate * (
        0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def synchronize_gradients(
    parameters,
    world_size: int,
    *,
    bucket_elements: int = 16_000_000,
) -> None:
    """Average possibly-unused dense gradients in a fixed collective order.

    Folding examples from different tracks can exercise different parameter
    branches on different ranks. Conditional per-parameter collectives can
    therefore deadlock. We first compute the union of locally active gradients
    and then reduce deterministic, dtype-homogeneous flat buckets. Parameters
    unused by every rank remain ``grad=None`` so AdamW does not decay them.
    """

    import torch
    import torch.distributed as dist

    parameters = list(parameters)
    if world_size < 1 or bucket_elements < 1:
        raise ValueError("world_size and bucket_elements must be positive")
    if not parameters:
        return
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("all synchronized parameters must be on one device")
    active = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.uint8,
    )
    dist.all_reduce(active, op=dist.ReduceOp.MAX)

    groups: dict[torch.dtype, list] = {}
    for index, parameter in enumerate(parameters):
        if active[index].item():
            if parameter.grad is not None and parameter.grad.is_sparse:
                raise ValueError("foldingtrain_edm_v1 requires dense gradients")
            groups.setdefault(parameter.dtype, []).append(parameter)

    for group in groups.values():
        bucket = []
        elements = 0

        def flush() -> None:
            nonlocal bucket, elements
            if not bucket:
                return
            tensors = [
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter)
                for parameter in bucket
            ]
            flat = torch._utils._flatten_dense_tensors(tensors)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            for parameter, reduced in zip(
                bucket,
                torch._utils._unflatten_dense_tensors(flat, tensors),
                strict=True,
            ):
                if parameter.grad is None:
                    parameter.grad = reduced
                else:
                    parameter.grad.copy_(reduced)
            bucket = []
            elements = 0

        for parameter in group:
            if bucket and elements + parameter.numel() > bucket_elements:
                flush()
            bucket.append(parameter)
            elements += parameter.numel()
        flush()

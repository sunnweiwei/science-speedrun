"""Noise construction, training losses, and cheap structure metrics."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def random_rotation(batch_size: int, device: torch.device) -> torch.Tensor:
    matrix = torch.randn(batch_size, 3, 3, device=device)
    q, r = torch.linalg.qr(matrix.float())
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q = q * sign
    determinant = torch.linalg.det(q)
    q[:, :, -1] *= determinant.unsqueeze(-1)
    return q


def prepare_noisy_batch(
    batch: dict[str, torch.Tensor],
    sigma_min: float = 0.02,
    sigma_max: float = 2.0,
) -> dict[str, torch.Tensor]:
    coords = batch["coords"].float()
    coord_mask = batch["coord_mask"]
    rotation = random_rotation(len(coords), coords.device)
    target = torch.matmul(coords, rotation)
    log_sigma = torch.empty(len(coords), device=coords.device).uniform_(
        math.log(sigma_min), math.log(sigma_max)
    )
    sigma = log_sigma.exp()
    noisy = target + torch.randn_like(target) * sigma[:, None, None]
    mask = coord_mask.unsqueeze(-1)
    denominator = coord_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
    noisy_center = (noisy * mask).sum(dim=1, keepdim=True) / denominator
    noisy = (noisy - noisy_center) * mask
    target_center = (target * mask).sum(dim=1, keepdim=True) / denominator
    target = (target - target_center) * mask
    output = dict(batch)
    output["target_coords"] = target
    output["noisy_coords"] = noisy
    output["sigma"] = sigma
    return output


def approximate_lddt(
    predicted: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor
) -> torch.Tensor:
    predicted_distance = torch.cdist(predicted.float(), predicted.float())
    target_distance = torch.cdist(target.float(), target.float())
    pair_mask = coord_mask.unsqueeze(2) & coord_mask.unsqueeze(1)
    eye = torch.eye(pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool)
    pair_mask = pair_mask & ~eye.unsqueeze(0) & (target_distance < 1.5)
    delta = (predicted_distance - target_distance).abs()
    score = sum((delta < threshold).float() for threshold in (0.05, 0.1, 0.2, 0.4)) / 4
    denominator = pair_mask.sum().clamp_min(1)
    return (score * pair_mask).sum() / denominator


def folding_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    predicted = outputs["pred_coords"].float()
    target = batch["target_coords"].float()
    coord_mask = batch["coord_mask"]
    squared_error = ((predicted - target) ** 2).sum(dim=-1)
    sigma_weight = 1.0 / (batch["sigma"] ** 2 + 0.25)
    coordinate_loss = (
        squared_error * coord_mask * sigma_weight.unsqueeze(-1)
    ).sum() / coord_mask.sum().clamp_min(1)

    predicted_distance = torch.cdist(predicted, predicted)
    target_distance = torch.cdist(target, target)
    pair_mask = coord_mask.unsqueeze(2) & coord_mask.unsqueeze(1)
    eye = torch.eye(pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool)
    pair_mask = pair_mask & ~eye.unsqueeze(0) & (target_distance < 3.2)
    distance_loss = F.smooth_l1_loss(
        predicted_distance[pair_mask], target_distance[pair_mask]
    )

    edges = torch.linspace(
        0.05,
        3.2,
        outputs["distogram_logits"].shape[-1] - 1,
        device=target.device,
    )
    bins = torch.bucketize(target_distance, edges)
    distogram_loss = F.cross_entropy(
        outputs["distogram_logits"][pair_mask].float(), bins[pair_mask]
    )
    total = coordinate_loss + 0.5 * distance_loss + 0.2 * distogram_loss
    rmse = torch.sqrt((squared_error * coord_mask).sum() / coord_mask.sum().clamp_min(1))
    lddt = approximate_lddt(predicted, target, coord_mask)
    return {
        "loss": total,
        "coordinate_loss": coordinate_loss,
        "distance_loss": distance_loss,
        "distogram_loss": distogram_loss,
        "coordinate_rmse_normalized": rmse,
        "approx_lddt": lddt,
    }

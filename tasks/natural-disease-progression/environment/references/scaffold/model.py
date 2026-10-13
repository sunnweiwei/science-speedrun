"""Neural-network construction seam shared by training and inference."""

from __future__ import annotations

import torch


def build_model(config: dict, metadata: dict) -> torch.nn.Module:
    """Build the submission network from plain configuration and metadata.

    Implement this function and the returned module's ``forward(batch)`` method.
    The forward result must contain ``disease_risk`` with shape ``[B, D, 13]`` and
    ``death_risk`` with shape ``[B, 13]``. Both are probabilities that are
    nondecreasing across horizons.
    """
    raise NotImplementedError(
        "Implement build_model(config, metadata) with the neural architecture under study"
    )

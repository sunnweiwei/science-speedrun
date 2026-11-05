"""A deliberately small, method-neutral dual encoder for the task contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class StarterModel(nn.Module):
    """Encode normalized atom histograms and score ligand-pocket similarity."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if min(config.input_dim, config.hidden_dim, config.embedding_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.config = config
        self.molecule = self._tower(config)
        self.pocket = self._tower(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    @staticmethod
    def _tower(config: ModelConfig) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def encode_molecule(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.molecule(values), dim=-1)

    def encode_pocket(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pocket(values), dim=-1)

    def forward(self, molecule: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * self.encode_pocket(pocket) @ self.encode_molecule(molecule).T

    def loss(self, molecule: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
        logits = self(molecule, pocket)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.T, labels))

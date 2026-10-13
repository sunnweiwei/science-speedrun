"""Fourier-feature DeepSets spectrum encoder used by the runnable starter."""

from math import ceil

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Fixed DreaMS-style Fourier features used by the MassSpecGym baseline."""

    def __init__(self, x_min: float = 1e-4, x_max: float = 1000.0):
        super().__init__()
        frequencies = [
            1 / (x_min * index)
            for index in range(2, ceil(1 / x_min), 2)
        ] + [
            1 / index
            for index in range(2, ceil(x_max))
        ]
        self.register_buffer("frequencies", torch.tensor(frequencies).reshape(1, -1))

    @property
    def output_size(self) -> int:
        return 2 * self.frequencies.shape[1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = 2 * torch.pi * values @ self.frequencies
        return torch.cat((torch.cos(encoded), torch.sin(encoded)), dim=-1)


def mlp(input_size: int, hidden_size: int, output_size: int, layers: int, dropout: float):
    if layers < 1:
        raise ValueError("layers must be positive")
    modules = []
    for index in range(layers):
        source = input_size if index == 0 else hidden_size
        target = output_size if index == layers - 1 else hidden_size
        modules.append(nn.Linear(source, target))
        if index != layers - 1:
            modules.append(nn.ReLU())
            if dropout:
                modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


class DeepSetsFingerprint(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        fingerprint_size: int = 4096,
        layers_per_mlp: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fourier = FourierFeatures()
        mz_size = int(0.8 * hidden_size)
        self.mz_projection = nn.Linear(self.fourier.output_size, mz_size)
        self.intensity_projection = nn.Linear(1, hidden_size - mz_size)
        self.phi = mlp(hidden_size, hidden_size, hidden_size, layers_per_mlp, dropout)
        self.rho = mlp(
            hidden_size, hidden_size, fingerprint_size, layers_per_mlp, dropout
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        # The largest fixed frequency is 5,000, so a 1,000-Da peak produces
        # an angle of roughly 3e7.  Computing that matrix product in fp16
        # overflows before sin/cos are applied.  Keep only the deterministic
        # Fourier map in fp32; the learned projections still use the caller's
        # autocast policy.
        with torch.autocast(device_type=spectra.device.type, enabled=False):
            mz_features = self.fourier(spectra[..., :1].float())
        mz = self.mz_projection(mz_features)
        intensity = self.intensity_projection(spectra[..., 1:2])
        peaks = self.phi(torch.cat((mz, intensity), dim=-1))
        return torch.sigmoid(self.rho(peaks.sum(dim=-2)))

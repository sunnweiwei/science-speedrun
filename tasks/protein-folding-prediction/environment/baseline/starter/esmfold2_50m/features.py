"""Portable decoder for the task's canonical ESMC-6B INT8 input feature."""

from pathlib import Path

import torch
from safetensors.torch import load_file


WIDTH = 2560
GROUP_SIZE = 64
GROUPS = WIDTH // GROUP_SIZE


def load_esmc_final(path: str | Path) -> torch.Tensor:
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"values", "scales"}:
        raise ValueError("Expected only values and scales tensors")
    values, scales = payload["values"], payload["scales"]
    if values.dtype != torch.int8 or values.ndim != 2 or values.shape[1] != WIDTH:
        raise ValueError(f"Invalid INT8 ESMC values: {values.shape} {values.dtype}")
    if scales.dtype != torch.bfloat16 or scales.shape != (values.shape[0], GROUPS):
        raise ValueError(f"Invalid ESMC scales: {scales.shape} {scales.dtype}")
    groups = values.float().reshape(values.shape[0], GROUPS, GROUP_SIZE)
    return (groups * scales.float().unsqueeze(-1)).reshape(values.shape[0], WIDTH)

from pathlib import Path
import sys

import torch

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))
sys.modules.pop("data", None)
sys.modules.pop("model", None)

from data import ELEMENTS, atom_histogram  # noqa: E402
from model import ModelConfig, StarterModel  # noqa: E402


def test_minimal_starter_forward_and_backward():
    molecule = torch.stack([atom_histogram(["C", "C", "O"]),
                            atom_histogram(["N", "C", "S"])])
    pocket = torch.stack([atom_histogram(["C", "N", "O", "ZN"]),
                          atom_histogram(["C", "C", "C", "O"])])
    model = StarterModel(ModelConfig(input_dim=len(ELEMENTS), hidden_dim=16,
                                     embedding_dim=8))
    loss = model.loss(molecule, pocket)
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000

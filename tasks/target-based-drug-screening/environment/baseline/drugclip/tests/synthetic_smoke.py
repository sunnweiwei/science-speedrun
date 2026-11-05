#!/usr/bin/env python3
"""One synthetic optimizer step through the complete compact model."""

from pathlib import Path
import sys

import numpy as np
import torch

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))

from data import VOCABULARY, collate_pairs, structure_features  # noqa: E402
from model import CompactDrugCLIP, parameter_count, preset_config  # noqa: E402


def item(index: int) -> dict:
    molecule_atoms = ["C", "C", "N", "O", "H"]
    pocket_atoms = ["C", "N", "O", "S", "C", "H"]
    generator = np.random.default_rng(index)
    return {
        "molecule": structure_features(
            molecule_atoms, generator.normal(size=(5, 3)), max_atoms=16, seed=index
        ),
        "pocket": structure_features(
            pocket_atoms, generator.normal(size=(6, 3)), max_atoms=16, seed=index
        ),
        "molecule_id": f"mol-{index}", "pocket_id": f"pocket-{index}",
    }


def main() -> None:
    torch.manual_seed(0)
    config = preset_config(
        "tiny", len(VOCABULARY), encoder_width=64, encoder_layers=1,
        attention_heads=4, distance_basis=16, dropout=0.0,
        max_molecule_atoms=16, max_pocket_atoms=16,
        gradient_checkpointing=False,
    )
    model = CompactDrugCLIP(config)
    batch = collate_pairs([item(index) for index in range(4)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(batch["molecule"], batch["pocket"])
    loss = model.contrastive_loss(logits, batch["molecule_ids"], batch["pocket_ids"])
    if logits.shape != (4, 4) or not torch.isfinite(loss):
        raise RuntimeError("invalid synthetic forward pass")
    loss.backward()
    optimizer.step()
    print({"status": "PASS", "parameters": parameter_count(model),
           "loss": float(loss.detach())})


if __name__ == "__main__":
    main()

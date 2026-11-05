from pathlib import Path
import sys

import numpy as np
import torch

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))
sys.modules.pop("data", None)
sys.modules.pop("model", None)

from data import VOCABULARY, collate_pairs, structure_features  # noqa: E402
from model import CompactDrugCLIP, ModelConfig, preset_config  # noqa: E402


def test_legacy_scratch_preset_has_paper_dimensions():
    config = preset_config("drugclip-original", len(VOCABULARY))
    assert config.encoder_width == 512
    assert config.encoder_layers == 15
    assert config.attention_heads == 64
    assert int(config.encoder_width * config.ffn_multiplier) == 2048
    assert config.distance_basis == 128


def test_bare_legacy_model_config_defaults_to_paper_dimensions():
    config = ModelConfig(vocab_size=len(VOCABULARY))
    assert (config.encoder_width, config.encoder_layers,
            config.attention_heads) == (512, 15, 64)


def test_hydrogen_removal_padding_forward_and_backward():
    config = preset_config(
        "tiny", len(VOCABULARY), encoder_width=64, encoder_layers=1,
        attention_heads=4, distance_basis=16, dropout=0.0,
        max_molecule_atoms=8, max_pocket_atoms=8,
        gradient_checkpointing=False,
    )
    items = []
    for index, length in enumerate((4, 6, 5)):
        atoms = ["C"] * length + ["H"]
        coords = np.arange((length + 1) * 3, dtype=np.float32).reshape(-1, 3)
        items.append({
            # Official benchmark LMDBs commonly store coordinates as nested
            # Python lists while processed training LMDBs store ndarrays.
            "molecule": structure_features(atoms, coords.tolist(), max_atoms=8, seed=index),
            "pocket": structure_features(atoms, coords, max_atoms=8, seed=index),
            "molecule_id": f"m{index}", "pocket_id": f"p{index}",
        })
    batch = collate_pairs(items)
    assert batch["molecule"]["tokens"].shape == (3, 8)
    model = CompactDrugCLIP(config)
    logits = model(batch["molecule"], batch["pocket"])
    loss = model.contrastive_loss(logits, batch["molecule_ids"], batch["pocket_ids"])
    loss.backward()
    assert logits.shape == (3, 3)
    assert torch.isfinite(loss)

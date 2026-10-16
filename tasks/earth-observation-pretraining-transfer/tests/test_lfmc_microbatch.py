"""Microbatch weighting equals official whole-batch masked MSE and gradients."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from lfmc_model import LFMCModel, backward_microbatches
from train_lfmc import evaluate
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig


class TestMicrobatch(unittest.TestCase):
    def test_unequal_masks_match_whole_batch(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = EncoderConfig(
                supported_modality_names=["sentinel2_l2a", "sentinel1"],
                embedding_size=32,
                num_heads=4,
                depth=1,
                drop_path=0,
            )
            torch.save(config.build().state_dict(), root / "encoder.pt")
            (root / "config.json").write_text(json.dumps(config.as_dict()))
            with (root / "encoder.pt").open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            (root / "provenance.json").write_text(
                json.dumps(
                    {"origin": "own_from_scratch_training", "encoder_sha256": sha}
                )
            )
            model = LFMCModel(str(root))
            other = copy.deepcopy(model)
            batch = {
                "ids": ["a", "b", "c"],
                "pixels": {
                    "sentinel2_l2a": torch.randn(3, 8, 8, 2, 12),
                    "sentinel1": torch.randn(3, 8, 8, 2, 2),
                },
                "valid": {
                    "sentinel2_l2a": torch.ones(3, 2, 3, dtype=torch.bool),
                    "sentinel1": torch.ones(3, 2, 1, dtype=torch.bool),
                },
                "target": torch.randn(3, 8, 8),
                "target_valid": torch.zeros(3, 8, 8, dtype=torch.bool),
            }
            batch["target_valid"][0, 4, 4] = True
            batch["target_valid"][1, 2:4, 2:4] = True
            result = model(
                batch["pixels"], batch["valid"], batch["target"], batch["target_valid"]
            )
            loss = result.loss_dict["regress"]
            loss.backward()
            micro, count = backward_microbatches(
                other, batch, torch.device("cpu"), microbatch=1
            )
            self.assertEqual(count, 5)
            self.assertAlmostEqual(float(loss.detach()), micro, places=5)
            squared = 0.0
            denom = 0.0
            for a, b in zip(model.parameters(), other.parameters()):
                self.assertEqual(a.grad is None, b.grad is None)
                if a.grad is not None:
                    squared += float((a.grad - b.grad).double().square().sum())
                    denom += float(a.grad.double().square().sum())
            self.assertLess((squared / denom) ** 0.5, 2e-5)
            metrics = evaluate(other, [batch], torch.device("cpu"), microbatch=1)
            self.assertEqual(metrics["windows"], 3)
            self.assertEqual(metrics["valid_pixels"], 5)
            self.assertAlmostEqual(metrics["mse"], micro, places=5)
            other.zero_grad(set_to_none=True)
            batch["target_valid"].zero_()
            empty, count = backward_microbatches(
                other, batch, torch.device("cpu"), microbatch=1
            )
            self.assertEqual(empty, 0)
            self.assertEqual(count, 0)
            self.assertTrue(
                all(p.grad is None or not p.grad.any() for p in other.parameters())
            )


if __name__ == "__main__":
    unittest.main()

"""Missing-observation spatial features preserve batch independence and gradients."""

from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from masked_spatial import spatial_features
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig


class TestSpatial(unittest.TestCase):
    def test_missing_batches_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = (
            EncoderConfig(
                supported_modality_names=["sentinel2_l2a", "sentinel1"],
                embedding_size=32,
                num_heads=4,
                depth=1,
                drop_path=0,
            )
            .build()
            .eval()
        )
        pixels = {
            "sentinel2_l2a": torch.randn(3, 8, 8, 2, 12),
            "sentinel1": torch.randn(3, 8, 8, 2, 2),
        }
        valid = {
            "sentinel2_l2a": torch.ones(3, 2, 3, dtype=torch.bool),
            "sentinel1": torch.ones(3, 2, 1, dtype=torch.bool),
        }
        valid["sentinel2_l2a"][0, 1] = False
        valid["sentinel1"][1] = False
        for v in valid.values():
            v[2] = False
        features = spatial_features(model, pixels, valid)
        self.assertEqual(tuple(features.shape), (3, 32, 2, 2))
        self.assertTrue(torch.equal(features[2], torch.zeros_like(features[2])))
        for i in range(3):
            alone = spatial_features(
                model,
                {k: v[i : i + 1] for k, v in pixels.items()},
                {k: v[i : i + 1] for k, v in valid.items()},
            )
            torch.testing.assert_close(features[i : i + 1], alone, atol=2e-6, rtol=2e-6)
        changed = {k: v.clone() for k, v in pixels.items()}
        changed["sentinel2_l2a"][0, :, :, 1, :] = 999
        changed["sentinel1"][1] = 999
        torch.testing.assert_close(
            features, spatial_features(model, changed, valid), atol=0, rtol=0
        )
        model.train()
        spatial_features(model, pixels, valid).square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(
            grads
            and all(torch.isfinite(g).all() for g in grads)
            and any(torch.any(g != 0) for g in grads)
        )
        with self.assertRaises(ValueError):
            spatial_features(model, pixels, {"sentinel1": valid["sentinel1"]})


if __name__ == "__main__":
    unittest.main()

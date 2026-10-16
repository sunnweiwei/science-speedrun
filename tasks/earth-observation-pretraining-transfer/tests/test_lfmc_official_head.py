"""Official LFMC UNet/loss consume differentiable masked spatial features."""

from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from masked_spatial import spatial_features
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from rslearn.models.component import FeatureMaps
from rslearn.models.unet import UNetDecoder
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.per_pixel_regression import PerPixelRegressionHead


class TestHead(unittest.TestCase):
    def test_official_unet_masked_loss_and_backprop(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        encoder = EncoderConfig(
            supported_modality_names=["sentinel2_l2a", "sentinel1"],
            embedding_size=32,
            num_heads=4,
            depth=1,
            drop_path=0,
        ).build()
        # Preserve official decoder widths/depth; only fixture encoder width is small.
        decoder = UNetDecoder(
            in_channels=[[4, 32]],
            out_channels=1,
            conv_layers_per_resolution=2,
            num_channels={8: 512, 4: 512, 2: 256, 1: 128},
        )
        head = PerPixelRegressionHead()
        pixels = {
            "sentinel2_l2a": torch.randn(2, 8, 8, 2, 12),
            "sentinel1": torch.randn(2, 8, 8, 2, 2),
        }
        valid = {
            "sentinel2_l2a": torch.ones(2, 2, 3, dtype=torch.bool),
            "sentinel1": torch.ones(2, 2, 1, dtype=torch.bool),
        }
        for v in valid.values():
            v[1] = False
        features = spatial_features(encoder, pixels, valid)
        context = ModelContext(inputs=[], metadatas=[])
        logits = decoder(FeatureMaps([features]), context)
        self.assertEqual(tuple(logits.feature_maps[0].shape), (2, 1, 8, 8))
        targets = []
        for value in [40.0, 70.0]:
            y = torch.full((1, 1, 8, 8), -1.0)
            y[0, 0, 4, 4] = value
            mask = torch.zeros_like(y, dtype=torch.bool)
            mask[0, 0, 4, 4] = True
            targets.append({"values": RasterImage(y), "valid": RasterImage(mask)})
        result = head(logits, context, targets)
        expected = (
            (logits.feature_maps[0][:, 0, 4, 4] - torch.tensor([40.0, 70.0])) ** 2
        ).mean()
        torch.testing.assert_close(result.loss_dict["regress"], expected)
        result.loss_dict["regress"].backward()
        for module in [encoder, decoder]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(
                grads
                and all(torch.isfinite(g).all() for g in grads)
                and any(torch.any(g != 0) for g in grads)
            )


if __name__ == "__main__":
    unittest.main()

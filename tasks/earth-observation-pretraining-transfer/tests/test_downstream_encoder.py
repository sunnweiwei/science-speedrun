"""Portable wrapper preserves weights/features and enables encoder gradients."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from portable_encoder import PortableAWFEncoder
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.pooling import pool_unmasked_tokens, PoolingType
from rslearn.train.model_context import ModelContext, RasterImage


class TestAdapter(unittest.TestCase):
    def test_exact_features_gradients_and_provenance(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = EncoderConfig(
                supported_modality_names=["sentinel2_l2a"],
                embedding_size=32,
                num_heads=4,
                depth=1,
                drop_path=0.0,
            )
            model = config.build()
            torch.save(model.state_dict(), root / "encoder.pt")
            (root / "config.json").write_text(json.dumps(config.as_dict()))
            with (root / "encoder.pt").open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            provenance = {"origin": "own_from_scratch_training", "encoder_sha256": sha}
            (root / "provenance.json").write_text(json.dumps(provenance))
            adapter = PortableAWFEncoder(str(root)).eval()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, adapter.model.state_dict()[key]))
            context = ModelContext(
                inputs=[{"sentinel2_l2a": RasterImage(torch.randn(12, 1, 8, 8))}],
                metadatas=[],
            )
            sample, _, _ = adapter._prepare_modality_inputs_legacy(context)
            model.eval()
            expected = pool_unmasked_tokens(
                model(sample, patch_size=4, fast_pass=True)["tokens_and_masks"],
                PoolingType.MEAN,
                spatial_pooling=True,
            ).permute(0, 3, 1, 2)
            actual = adapter(context).feature_maps[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            adapter.train()
            adapter(context).feature_maps[0].square().mean().backward()
            self.assertTrue(
                any(
                    p.grad is not None and torch.any(p.grad != 0)
                    for p in adapter.model.parameters()
                )
            )
            provenance["origin"] = "official_public_pretrained_reference"
            (root / "provenance.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "Ineligible"):
                PortableAWFEncoder(str(root))


if __name__ == "__main__":
    unittest.main()

"""Exact exported encoder inside the official rslearn feature-extractor interface."""

import hashlib
import json
from pathlib import Path
from types import MethodType
import torch
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.pooling import pool_unmasked_tokens, PoolingType
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.olmoearth_pretrain.model import OlmoEarth, TOKENS_IN_BATCH_KEY


def keep_padding_mask(self, new_mask, fast_pass):
    return None if fast_pass else new_mask


class PortableAWFEncoder(OlmoEarth):
    """AWF-only adapter: official inputs/pooling, strict local encoder load, no hub."""

    def __init__(self, package: str):
        FeatureExtractor.__init__(self)
        root = Path(package)
        provenance = json.loads((root / "provenance.json").read_text())
        if (
            provenance.get("training_validity") == "invalid"
            or provenance.get("origin") == "official_public_pretrained_reference"
        ):
            raise ValueError("Ineligible encoder")
        with (root / "encoder.pt").open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != provenance["encoder_sha256"]:
            raise ValueError("Encoder digest mismatch")
        config = json.loads((root / "config.json").read_text())
        self.model = EncoderConfig.from_dict(config).build()
        self.model.load_state_dict(
            torch.load(root / "encoder.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model._maybe_get_attn_mask = MethodType(keep_padding_mask, self.model)
        self.patch_size = 4
        self.embedding_size = config["embedding_size"]
        self.autocast_dtype = torch.bfloat16
        self.forward_kwargs = {}
        self.token_pooling = True
        self.use_legacy_timestamps = True
        self.encoder_sha256 = actual

    def forward(self, context):
        sample, modalities, device = self._prepare_modality_inputs_legacy(context)
        if modalities != ["sentinel2_l2a"]:
            raise ValueError(f"AWF-only adapter received {modalities}")
        # Released AWF coverage is complete. Fail rather than silently changing
        # missing-modality semantics; LFMC needs its own explicit adapter.
        if torch.any(sample.sentinel2_l2a_mask != 0):
            raise ValueError("Unexpected missing AWF observations")
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            tokens = self.model(sample, patch_size=4, fast_pass=True)[
                "tokens_and_masks"
            ]
            pooled = pool_unmasked_tokens(
                tokens, PoolingType.MEAN, spatial_pooling=True
            )
        context.context_dict[TOKENS_IN_BATCH_KEY] = self.compute_tokens_in_batch(
            tokens, modalities
        )
        return FeatureMaps([pooled.permute(0, 3, 1, 2).float()])

"""Differentiable spatial features with explicit temporal/band observation masks.

This is the missing-observation bridge for downstream heads, not a new encoder.
Inputs retain the pretraining HWT-band layout and supplied observation metadata.
"""

from types import MethodType
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens


def keep_mask(self, new_mask, fast_pass):
    return None if fast_pass else new_mask


def spatial_features(model, pixels, valid, patch_size=4):
    """Return BCHW spatial features without dropping all-missing samples.

    pixels: normalized floating tensors [B,H,W,T,C].
    valid: boolean tensors [B,T,S] for every supplied modality, S=band groups.
    No missingness is inferred from a pixel value. Caller controls train/eval mode.
    """
    if not pixels or set(pixels) != set(valid):
        raise ValueError("Every modality requires explicit observation validity")
    first = next(iter(pixels.values()))
    if first.ndim != 5:
        raise ValueError("Expected BHWTC pixels")
    b, h, w, t, _ = first.shape
    if h % patch_size or w % patch_size:
        raise ValueError("Spatial size must be divisible by patch size")
    for name, x in pixels.items():
        expected = (b, t, Modality.get(name).num_band_sets)
        if (
            x.shape[:4] != first.shape[:4]
            or x.device != first.device
            or valid[name].shape != expected
            or valid[name].dtype != torch.bool
            or valid[name].device != first.device
        ):
            raise ValueError("Inconsistent modality shape/device or validity")
    present = torch.stack([v.flatten(1).any(1) for v in valid.values()], dim=1)
    observed = present.any(1)
    output = first.new_zeros(
        (b, model.embedding_size, h // patch_size, w // patch_size), dtype=torch.float32
    )
    if not observed.any():
        return output
    index = observed.nonzero(as_tuple=True)[0]
    kwargs = {}
    included = []
    all_valid = True
    for name, x in pixels.items():
        v = valid[name][index]
        if not v.any():
            continue
        included.append(name)
        kwargs[name] = x[index]
        mask = torch.zeros(
            (len(index), h, w, t, v.shape[-1]), device=x.device, dtype=torch.int32
        )
        mask.masked_fill_(~v[:, None, None, :, :], 3)
        kwargs[name + "_mask"] = mask
        all_valid = all_valid and bool(v.all())
    timestamp = torch.zeros((len(index), t, 3), device=first.device, dtype=torch.int32)
    timestamp[:, :, 0] = 1
    timestamp[:, :, 1] = torch.arange(t, device=first.device)
    timestamp[:, :, 2] = 2024
    kwargs["timestamps"] = timestamp
    model._maybe_get_attn_mask = MethodType(keep_mask, model)
    with torch.autocast(
        device_type=first.device.type,
        dtype=torch.bfloat16,
        enabled=first.device.type == "cuda",
    ):
        tokens = model(
            MaskedOlmoEarthSample(**kwargs), patch_size=patch_size, fast_pass=all_valid
        )["tokens_and_masks"]
        pooled = pool_unmasked_tokens(tokens, PoolingType.MEAN, spatial_pooling=True)
        observed_modalities = torch.stack(
            [valid[n][index].flatten(1).any(1) for n in included], dim=1
        ).sum(1)
        pooled = pooled * (len(included) / observed_modalities)[:, None, None, None]
    return output.index_copy(0, index, pooled.permute(0, 3, 1, 2).float())

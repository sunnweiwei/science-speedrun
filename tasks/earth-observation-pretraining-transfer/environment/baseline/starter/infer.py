"""Portable frozen OlmoEarth encoder. Input HDF5 has pixels and IDs only."""

import argparse
import json
import os
from pathlib import Path
from types import MethodType
import h5py
import numpy as np
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.pooling import pool_unmasked_tokens, PoolingType


def inference_attention_mask(self, new_mask, fast_pass):
    # Upstream v1 discards this padding mask in eval mode, although missing
    # observations produce variable sequence lengths. Preserve the training
    # attention-mask semantics without enabling dropout or changing weights.
    return None if fast_pass else new_mask


def configure_inference(model):
    model.eval()
    model._maybe_get_attn_mask = MethodType(inference_attention_mask, model)
    return model


def normalized_pixels(h, start, end):
    normalizer = Normalizer(Strategy.COMPUTED)
    out = {}
    for name in ["sentinel2_l2a", "sentinel1"]:
        if name not in h:
            continue
        arr = h[name][start:end].astype(np.float32)
        if name == "sentinel1":
            arr = 10 * np.log10(np.maximum(arr, 1e-6))
        out[name] = normalizer.normalize(Modality.get(name), arr).astype(np.float32)
    return out


def embed(
    model, pixels, device, patch_size=4, center=False, valid=None, bilinear_point=False
):
    batch = len(next(iter(pixels.values())))
    observed = np.zeros(batch, dtype=bool)
    for name in pixels:
        observed |= (
            valid[name].reshape(batch, -1).any(axis=1)
            if valid is not None and name in valid
            else True
        )
    if not observed.all():
        # No source observations: an explicit no-information vector. Keep the row
        # in evaluation; its downstream prediction is determined by the fitted probe.
        result = np.zeros((batch, model.embedding_size), dtype=np.float32)
        if observed.any():
            result[observed] = embed(
                model,
                {k: v[observed] for k, v in pixels.items()},
                device,
                patch_size=patch_size,
                center=center,
                bilinear_point=bilinear_point,
                valid={k: v[observed] for k, v in valid.items()}
                if valid is not None
                else None,
            )
        return result
    kwargs = {}
    for name, arr in pixels.items():
        val = torch.from_numpy(arr).to(device)
        kwargs[name] = val
        spec = Modality.get(name)
        mask = torch.zeros(
            (*val.shape[:-1], spec.num_band_sets), dtype=torch.int32, device=device
        )
        if valid is not None and name in valid:
            missing = torch.from_numpy(~valid[name].astype(bool)).to(device)
            if missing.ndim == 2:
                missing = missing[:, :, None]
            mask.masked_fill_(missing[:, None, None, :, :], 3)
        kwargs[name + "_mask"] = mask
    b, _, _, t, _ = next(iter(pixels.values())).shape
    timestamps = torch.zeros((b, t, 3), dtype=torch.int32, device=device)
    timestamps[:, :, 0] = 1
    timestamps[:, :, 1] = torch.arange(t, device=device)
    timestamps[:, :, 2] = 2024
    kwargs["timestamps"] = timestamps
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        result = model(
            MaskedOlmoEarthSample(**kwargs),
            patch_size=patch_size,
            fast_pass=valid is None or all(np.all(v) for v in valid.values()),
        )["tokens_and_masks"]
        vec = pool_unmasked_tokens(result, PoolingType.MEAN, spatial_pooling=center)
        if center:
            if bilinear_point:
                # AWF's published head upsamples by four before its pointwise
                # linear classifier. Sampling here preserves that interpolation.
                vec = torch.nn.functional.interpolate(
                    vec.permute(0, 3, 1, 2),
                    scale_factor=patch_size,
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            vec = vec[:, vec.shape[1] // 2, vec.shape[2] // 2]
            if valid:
                present = np.stack(
                    [
                        valid[name].reshape(batch, -1).any(axis=1)
                        if name in valid
                        else np.ones(batch, dtype=bool)
                        for name in pixels
                    ],
                    axis=1,
                )
                # Upstream averages over modalities present anywhere in the
                # batch. Average over each sample's observed modalities instead.
                scale = present.any(axis=0).sum() / present.sum(axis=1)
                vec = (
                    vec
                    * torch.as_tensor(scale, device=device, dtype=vec.dtype)[:, None]
                )
    return vec.float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    if (
        a.device == "cuda"
        and os.environ.get("EARTH_SLURM_PARTITION")
        and os.environ.get("SLURM_JOB_PARTITION") != os.environ["EARTH_SLURM_PARTITION"]
    ):
        raise RuntimeError("GPU execution requires Slurm configured-gpu-partition")
    config = json.loads((a.model / "config.json").read_text())
    model = EncoderConfig.from_dict(config).build()
    model.load_state_dict(
        torch.load(a.model / "encoder.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    configure_inference(model).to(a.device)
    with h5py.File(a.input, "r") as h:
        ids = h["ids"].asstr()[:]
        if h.attrs["task"] not in ("awf", "lfmc"):
            raise ValueError("Unknown evaluation task")
        bilinear_point = h.attrs["task"] == "awf"
        chunks = []
        for start in range(0, len(ids), a.batch_size):
            valid = {
                name: h[name + "_valid"][start : start + a.batch_size]
                for name in ["sentinel2_l2a", "sentinel1"]
                if name + "_valid" in h
            }
            chunks.append(
                embed(
                    model,
                    normalized_pixels(h, start, start + a.batch_size),
                    torch.device(a.device),
                    center=True,
                    bilinear_point=bilinear_point,
                    valid=valid,
                )
            )
            if start % (a.batch_size * 50) == 0:
                print("embedded", start, len(ids), flush=True)
    features = np.concatenate(chunks)
    if (
        features.shape != (len(ids), config["embedding_size"])
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid embeddings")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, ids=np.asarray(ids, dtype=str), features=features)
    print(
        json.dumps(
            {"rows": len(ids), "dimension": features.shape[1], "output": str(a.output)}
        )
    )


if __name__ == "__main__":
    main()

"""CPU regression checks for missing observations and batch independence."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.model.resolve()))
from infer import EncoderConfig, Modality, configure_inference, embed

model = EncoderConfig.from_dict(
    json.loads((a.model / "config.json").read_text())
).build()
model.load_state_dict(
    torch.load(a.model / "encoder.pt", map_location="cpu", weights_only=True)
)
configure_inference(model)
rng = np.random.default_rng(42)
pixels = {
    "sentinel2_l2a": rng.normal(size=(3, 8, 8, 3, 12)).astype(np.float32),
    "sentinel1": rng.normal(size=(3, 8, 8, 3, 2)).astype(np.float32),
}
valid = {
    "sentinel2_l2a": np.ones((3, 3, 3), dtype=bool),
    "sentinel1": np.ones((3, 3, 1), dtype=bool),
}
valid["sentinel1"][0] = False
valid["sentinel2_l2a"][1, 1, 1] = False
for mask in valid.values():
    mask[2] = False


def run(x, masks):
    return embed(model, x, torch.device("cpu"), center=True, valid=masks)


joint = run(pixels, valid)
alone = np.concatenate(
    [
        run(
            {k: v[i : i + 1] for k, v in pixels.items()},
            {k: v[i : i + 1] for k, v in valid.items()},
        )
        for i in range(3)
    ]
)
altered = {k: v.copy() for k, v in pixels.items()}
for name in altered:
    spec = Modality.get(name)
    for row, time, group in np.argwhere(~valid[name]):
        for band in spec.band_sets[group].bands:
            altered[name][row, :, :, time, spec.band_order.index(band)] = 100000
perturbed = run(altered, valid)
report = {
    "single_vs_batch_max_abs": float(np.max(abs(joint - alone))),
    "masked_pixel_perturbation_max_abs": float(np.max(abs(joint - perturbed))),
    "unobserved_row_is_zero": bool(np.all(joint[2] == 0)),
}
assert report["single_vs_batch_max_abs"] < 1e-5, report
assert report["masked_pixel_perturbation_max_abs"] < 1e-5, report
assert report["unobserved_row_is_zero"], report
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

"""Seasonal band means/stds: no learned representation, no labels."""

import argparse
from pathlib import Path
import sys
import h5py
import numpy as np

for _starter in ("baseline/starter", "starter"):  # repository / workspace layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _starter))
from infer import normalized_pixels, Modality

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
features = []
with h5py.File(a.input, "r") as h:
    ids = h["ids"].asstr()[:]
    for i in range(0, len(ids), 64):
        pixels = normalized_pixels(h, i, i + 64)
        pieces = []
        for name, arr in pixels.items():
            if name + "_valid" in h:
                valid = h[name + "_valid"][i : i + 64].astype(bool)
                if valid.ndim == 2:
                    valid = np.repeat(
                        valid[:, :, None], Modality.get(name).num_band_sets, axis=2
                    )
                for j in range(len(arr)):
                    for g, bandset in enumerate(Modality.get(name).band_sets):
                        bands = [
                            Modality.get(name).band_order.index(b)
                            for b in bandset.bands
                        ]
                        good = np.flatnonzero(valid[j, :, g])
                        for t in np.flatnonzero(~valid[j, :, g]):
                            for band in bands:
                                arr[j, :, :, t, band] = (
                                    arr[j, :, :, good[np.argmin(abs(good - t))], band]
                                    if len(good)
                                    else 0
                                )
            pieces.extend(
                [
                    arr.mean(axis=(1, 2)).reshape(len(arr), -1),
                    arr.std(axis=(1, 2)).reshape(len(arr), -1),
                ]
            )
        features.append(np.concatenate(pieces, axis=1))
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.output, ids=np.asarray(ids, dtype=str), features=np.concatenate(features))

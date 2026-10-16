"""Audit data coverage and exact site separation without fitting a model."""

import argparse
from pathlib import Path
import json
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
report = {}
for task in ["awf", "lfmc"]:
    rows = json.loads((a.data / task / "labels.json").read_text())
    out = {}
    sites = {}
    for split in ["train", "val", "test"]:
        sites[split] = {
            tuple(round(x, 6) for x in r["latlon"]) for r in rows if r["split"] == split
        }
    out["exact_pixel_site_overlap"] = {
        x + "_" + y: len(sites[x] & sites[y])
        for x, y in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    with h5py.File(a.data / task / "inputs.h5") as h:
        ids = h["ids"].asstr()[:]
        assert ids.tolist() == [r["id"] for r in rows]
        present = np.zeros(len(ids), bool)
        for mod in ["sentinel2_l2a", "sentinel1"]:
            if mod not in h:
                continue
            valid = (
                h[mod + "_valid"][:].astype(bool)
                if mod + "_valid" in h
                else np.ones((len(ids), 12, 3 if mod == "sentinel2_l2a" else 1), bool)
            )
            present |= valid.reshape(len(valid), -1).any(axis=1)
            out[mod] = {
                "shape": list(h[mod].shape),
                "availability_shape": list(valid.shape),
                "missing_groups": int((~valid).sum()),
            }
        out["no_observations"] = int((~present).sum())
        out["rows"] = len(ids)
        out["no_observations_by_split"] = {
            split: sum(
                not present[i] for i, r in enumerate(rows) if r["split"] == split
            )
            for split in ["train", "val", "test"]
        }
        out["no_observations_by_split"] = {
            k: int(v) for k, v in out["no_observations_by_split"].items()
        }
    report[task] = out
    assert not any(out["exact_pixel_site_overlap"].values()), (
        "Public split has an exact site overlap"
    )
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

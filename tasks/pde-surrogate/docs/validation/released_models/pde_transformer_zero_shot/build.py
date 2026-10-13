#!/usr/bin/env python3
"""Bundle per-dataset normalization buffers from the corpus files.

    python3 build.py --data DATA_DIR --source ape2d-full --out normalization.json
Reads norm_fields_sca_mean / norm_fields_sca_std (the buffers upstream's
MeanStdNorm uses) from each training file and stores them per dataset name.
"""
import argparse
import json
from pathlib import Path

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--source", default="ape2d-full"); ap.add_argument("--out", default="normalization.json")
a = ap.parse_args()
root = Path(a.data) / a.source
out = {}
for f in sorted(root.rglob("*.hdf5")):
    name = f.stem
    if name.endswith("_test"):
        continue
    with h5py.File(f, "r") as h:
        out[f"ape2d/{name}"] = {"pde": str(h["sims"].attrs["PDE"]),
                               "mean": [float(x) for x in h["norm_fields_sca_mean"][()].reshape(-1)],
                               "std": [float(x) for x in h["norm_fields_sca_std"][()].reshape(-1)]}
Path(a.out).write_text(json.dumps(out, indent=1))
print(f"{len(out)} datasets -> {a.out}")

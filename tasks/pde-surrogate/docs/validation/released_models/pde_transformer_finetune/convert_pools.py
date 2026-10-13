#!/usr/bin/env python3
"""Convert the task's split pools into PDE-Transformer's PBDL file format (one file per split).

    python3 convert_pools.py --data DATA_DIR --manifest split.json --dataset well/turbulent_radiative_layer_2D --out OUT_DIR

Writes OUT_DIR/<name>_train.hdf5, <name>_valid.hdf5 and <name>_test.hdf5 with
`sims/sim<i>` arrays (T, C, X, Y) float32 and the metadata dictionary upstream's
`download_setups_2d.py` attaches, so upstream's `2D_WELL` data type loads them.
Held-out test trajectories come from the manifest's `cases` (frames from `start`),
train/valid from its `pools`. Deterministic; no shuffling, no truncation.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from pde_io import family_of, read_window, trajectory_length  # noqa: E402

# upstream metadata (pdetransformer/data/download_well/download_setups_2d.py), Well channel order
SETUPS = {
    "well/turbulent_radiative_layer_2D": {
        "PDE": "Turbulent Radiative Layer", "Dimension": 2, "Fields Scheme": "dpVV",
        "Fields": ["Density", "Pressure", "Velocity X", "Velocity Y"], "Domain Extent": [1, 3],
        "Resolution": [128, 384], "Time Steps": 101, "Dt": 1.597033,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "open", "open"], "Constants": ["Cooling Time"],
        "_const_keys": ["tcool"]},
    "well/active_matter": {
        "PDE": "Active Matter", "Dimension": 2, "Fields Scheme": "cVVOOOOSSSS",
        "Fields": ["Concentration", "Velocity X", "Velocity Y", "Orientation XX", "Orientation XY", "Orientation YX",
                   "Orientation YY", "Strain XX", "Strain XY", "Strain YX", "Strain YY"],
        "Domain Extent": [10, 10], "Resolution": [256, 256], "Time Steps": 81, "Dt": 0.25,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic"] * 4,
        "Constants": ["Domain Extent", "Particle Alignment Strength", "Active Dipol Strength"],
        "_const_keys": ["L", "zeta", "alpha"]},
    "pdearena/navier_stokes_2d": {
        "PDE": "Navier-Stokes: Incompressible Cylinder Flow", "Dimension": 2, "Fields Scheme": "dVV",
        "Fields": ["Density", "Velocity X", "Velocity Y"], "Domain Extent": [32, 32], "Resolution": [128, 128],
        "Time Steps": 14, "Dt": 1.5, "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["wall"] * 4, "Constants": ["Buoyancy"], "_const_keys": ["buoyancy_y"]},
}


def write_split(path, entries, setup, data_dir, context, n_out):
    with h5py.File(path, "w") as f:
        g = f.create_group("sims", track_order=True)
        for k, v in setup.items():
            if not k.startswith("_"):
                g.attrs[k] = v
        for i, (fam, file, traj, start, n_frames) in enumerate(entries):
            frames, meta = read_window(fam, data_dir / file, traj, start, n_frames, 0)
            d = g.create_dataset(f"sim{i}", data=frames.astype(np.float32))
            for name, key in zip(setup["Constants"], setup["_const_keys"]):
                d.attrs[name] = float(meta["constants"].get(key, 0.0))
    return len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-train", type=int, default=0, help="cap on training trajectories (0 = all)")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text()); setup = SETUPS[args.dataset]
    data_dir, out = Path(args.data), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    name = args.dataset.split("/")[-1]
    ctx, k = man["context_frames"], None
    counts = {}
    for split in ("train", "valid"):
        entries = []
        for p in man["pools"][split]:
            if p["dataset"] != args.dataset:
                continue
            fam = family_of(p["source"]); file = Path(p["source"]) / p["file"]
            T = trajectory_length(fam, data_dir / file)
            entries += [(fam, file, t, 0, T) for t in range(*p["trajectories"])]
        if split == "train" and args.max_train:
            entries = entries[: args.max_train]
        counts[split] = write_split(out / f"{name}_{split}.hdf5", entries, setup, data_dir, ctx, 0)
    entries = []
    for c in man["cases"]:
        if c["dataset"] == args.dataset and c["split"] == "test":
            fam = family_of(c["source"]); file = Path(c["source"]) / c["file"]
            entries.append((fam, file, c["trajectory"], c["start"], ctx + c["n_steps_out"]))
    counts["test"] = write_split(out / f"{name}_test.hdf5", entries, setup, data_dir, ctx, 0)
    print(f"{args.dataset}: {counts} -> {out}")


if __name__ == "__main__":
    main()

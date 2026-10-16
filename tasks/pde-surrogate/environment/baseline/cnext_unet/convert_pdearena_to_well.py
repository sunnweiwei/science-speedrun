#!/usr/bin/env python3
"""Write the task's PDEArena split in The Well's HDF5 layout so The Well's trainer can train on it.

    python3 convert_pdearena_to_well.py --data /workspace/data --manifest /workspace/data/split.json \
        --out /workspace/research/cnext_unet/well-pdearena_navier_stokes_2d
--data may repeat; each shard is taken from the first directory that has it. Produces data/{train,valid}/*.hdf5
(one file per source shard, trajectories from the manifest's pools) plus stats.yaml (z-score statistics over the
training split). The held-out test shards are not in the workspace: their cases are skipped, and data/test/ is
written only when a --data directory holds them (a full materialization).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, action="append"); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-train-shards", type=int, default=0)
a = ap.parse_args()
man = json.loads(Path(a.manifest).read_text()); out = Path(a.out)
DS = "pdearena/navier_stokes_2d"
groups = defaultdict(list)   # (split, file) -> trajectory indices
for split in ("train", "valid"):
    for p in man["pools"][split]:
        if p["dataset"] == DS:
            groups[(split, p["file"])] += list(range(*p["trajectories"]))
for c in man["cases"]:
    if c["dataset"] == DS and c["split"] == "test":
        groups[("test", c["file"])].append(c["trajectory"])
if a.max_train_shards:
    tr = [k for k in groups if k[0] == "train"][a.max_train_shards:]
    for k in tr:
        del groups[k]
acc = {"u": [], "vx": [], "vy": []}; acc_d = {"u": [], "vx": [], "vy": []}
skipped = 0
for (split, file), trajs in sorted(groups.items()):
    src = next((Path(d) / "pdearena-ns" / file for d in a.data if (Path(d) / "pdearena-ns" / file).exists()), None)
    if src is None:
        if split == "test":
            skipped += 1; continue
        raise SystemExit(f"{split} shard {file} is in none of {a.data}")
    dst = out / "data" / split / (Path(file).stem + ".hdf5"); dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        s = f[next(iter(f.keys()))]; idx = sorted(trajs)
        u, vx, vy = (s[k][idx].astype(np.float32) for k in ("u", "vx", "vy"))
        t, x, y = s["t"][idx[0]].astype(np.float32), s["x"][idx[0]].astype(np.float32), s["y"][idx[0]].astype(np.float32)
        g.attrs["dataset_name"] = "pdearena_navier_stokes_2d"; g.attrs["grid_type"] = "cartesian"
        g.attrs["n_spatial_dims"] = 2; g.attrs["n_trajectories"] = len(idx); g.attrs["simulation_parameters"] = ["buoyancy"]
        g.attrs["buoyancy"] = float(s["buo_y"][idx[0]])
        t0 = g.create_group("t0_fields"); t0.attrs["field_names"] = ["u"]
        d = t0.create_dataset("u", data=u); d.attrs.update({"dim_varying": [True, True], "sample_varying": True, "time_varying": True})
        t1 = g.create_group("t1_fields"); t1.attrs["field_names"] = ["velocity"]
        d = t1.create_dataset("velocity", data=np.stack([vx, vy], axis=-1)); d.attrs.update({"dim_varying": [True, True], "sample_varying": True, "time_varying": True})
        t2 = g.create_group("t2_fields"); t2.attrs["field_names"] = []
        dims = g.create_group("dimensions"); dims.attrs["spatial_dims"] = ["x", "y"]
        for name, arr in (("time", t), ("x", x), ("y", y)):
            d = dims.create_dataset(name, data=arr); d.attrs.update({"sample_varying": False, "time_varying": name == "time"})
        bc = g.create_group("boundary_conditions")
        for name, dim, n in (("x_wall", "x", len(x)), ("y_wall", "y", len(y))):
            b = bc.create_group(name); m = np.zeros(n, bool); m[0] = m[-1] = True; b.create_dataset("mask", data=m)
            b.attrs.update({"associated_dims": [dim], "associated_fields": [], "bc_type": "WALL", "sample_varying": False, "time_varying": False})
        sc = g.create_group("scalars"); sc.attrs["field_names"] = ["buoyancy"]
        d = sc.create_dataset("buoyancy", data=np.float32(s["buo_y"][idx[0]])); d.attrs.update({"sample_varying": False, "time_varying": False})
        if split == "train":
            for k, arr in (("u", u), ("vx", vx), ("vy", vy)):
                acc[k].append(arr.reshape(-1)[::7]); acc_d[k].append(np.diff(arr, axis=1).reshape(-1)[::7])
    print(f"  {split:5s} {dst.name} n={len(idx)}")
def stats(v):
    v = np.concatenate(v).astype(np.float64); return float(v.mean()), float(v.std()), float(np.sqrt((v ** 2).mean()))
S = {k: stats(acc[k]) for k in acc}; SD = {k: stats(acc_d[k]) for k in acc_d}
def block(name, s):
    return f"  u: {s['u'][0]:.4E}\n  velocity:\n  - {s['vx'][0]:.4E}\n  - {s['vy'][0]:.4E}\n".replace("[0]", "")
def fmt(sel, src):
    return f"  u: {src['u'][sel]:.4E}\n  velocity:\n  - {src['vx'][sel]:.4E}\n  - {src['vy'][sel]:.4E}\n"
yaml = "mean:\n" + fmt(0, S) + "mean_delta:\n" + fmt(0, SD) + "rms:\n" + fmt(2, S) + "rms_delta:\n" + fmt(2, SD) + "std:\n" + fmt(1, S) + "std_delta:\n" + fmt(1, SD)
(out / "stats.yaml").write_text(yaml)
(out / "pdearena_navier_stokes_2d.yaml").write_text("dataset_name: pdearena_navier_stokes_2d\nn_spatial_dims: 2\nspatial_resolution:\n- 128\n- 128\nscalar_names: []\nconstant_scalar_names:\n- buoyancy\nfield_names:\n  0:\n  - u\n  1:\n  - velocity_x\n  - velocity_y\n  2: []\nconstant_field_names:\n  0: []\n  1: []\n  2: []\nboundary_condition_types:\n- WALL\ngrid_type: cartesian\n")
if skipped:
    print(f"skipped {skipped} held-out test shards that no --data directory holds")
print("stats:", {k: [round(x, 4) for x in v] for k, v in S.items()})

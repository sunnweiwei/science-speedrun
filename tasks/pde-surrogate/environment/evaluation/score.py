#!/usr/bin/env python3
"""Trusted grader for the pde-surrogate-rollout/v1 contract.

    python3 score.py --input INPUT_DIR --truth TRUTH_DIR --output OUTPUT_DIR --report report.json

Metric: The Well's VRMSE, per channel and rollout step,
    VRMSE = sqrt( mean_xy (pred - target)^2 / ( var_xy(target) + 1e-7 ) )
with the unbiased spatial variance (ddof=1), spatial reduction only, clamped to
VRMSE_MAX = 10. Aggregation: mean over channels, then steps, then cases of a
dataset; datasets are grouped (the pretraining corpus 'ape2d/*' is one group,
every other dataset is its own group) and the reward is the unweighted mean
over groups of a log-band map of the group's mean VRMSE:
0.01 -> 1.0, 0.1 -> 0.667, 1.0 -> 0.333, >= 10 -> 0.
Every case in the index stays in the denominator; a missing or invalid
prediction scores VRMSE_MAX everywhere and is listed with its reason.
"""
import argparse
import json
import math
import os
import stat
import sys
from pathlib import Path

import h5py
import numpy as np

CONTRACT = "pde-surrogate-rollout/v1"
VRMSE_MAX = 10.0
VRMSE_FLOOR = 0.01
EPS = 1e-7
WINDOWS = {"steps_1_5": (0, 5), "steps_6_12": (5, 12), "steps_13_30": (12, 30)}


def vrmse(pred, target):
    """(K, C) VRMSE per step and channel; inputs (K, C, X, Y) float64."""
    mse = ((pred - target) ** 2).mean(axis=(2, 3))
    var = target.var(axis=(2, 3), ddof=1)
    return np.minimum(np.sqrt(mse / (var + EPS)), VRMSE_MAX)


def nrmse_joint(pred, target):
    mse = ((pred - target) ** 2).mean(axis=(1, 2, 3))
    return np.minimum(np.sqrt(mse / ((target ** 2).mean(axis=(1, 2, 3)) + EPS)), VRMSE_MAX)


def reward_from_vrmse(v):
    lo, hi = math.log10(VRMSE_FLOOR), math.log10(VRMSE_MAX)
    return float(min(1.0, max(0.0, (hi - math.log10(max(v, VRMSE_FLOOR))) / (hi - lo))))


def open_package_h5(path, names):
    """Open an HDF5 file a package wrote, refusing anything but a regular file whose `names` are plain datasets.

    The grader reads with the verifier's privileges, outside every isolation mode, so it must not follow anything
    the package put in the file: an external or soft link, a virtual dataset or external raw storage can point at
    the held-out truth and would be resolved here. A FIFO or device in place of the file would block the grader."""
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("not a regular file (symlinks, FIFOs and devices are refused)")
    f = h5py.File(path, "r")
    try:
        for name in names:
            link = f.get(name, getlink=True)
            if link is None:
                raise KeyError(f"no {name!r} dataset")
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"{name!r} is a {type(link).__name__}; only plain datasets are read")
            d = f[name]
            if not isinstance(d, h5py.Dataset) or d.is_virtual or d.external:
                raise ValueError(f"{name!r} is not a plain dataset (virtual or external storage is refused)")
    except Exception:
        f.close()
        raise
    return f


def load_prediction(path: Path, shape):
    if not os.path.lexists(path):
        return None, "missing"
    try:
        with open_package_h5(path, ["prediction"]) as f:
            if f["prediction"].shape != tuple(shape):
                return None, f"wrong shape {tuple(f['prediction'].shape)} != {tuple(shape)}"
            arr = f["prediction"][()]
    except KeyError:
        return None, "no 'prediction' dataset"
    except Exception as exc:  # noqa: BLE001 - any read failure is a failed case
        return None, f"unreadable: {type(exc).__name__}: {str(exc)[:200]}"
    if arr.shape != tuple(shape):
        return None, f"wrong shape {tuple(arr.shape)} != {tuple(shape)}"
    if not np.issubdtype(arr.dtype, np.floating):
        return None, f"non-float dtype {arr.dtype}"
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        return None, "non-finite values"
    return arr, None


def score(input_dir: Path, truth_dir: Path, output_dir: Path):
    index = json.loads((input_dir / "index.json").read_text())
    if index.get("contract") != CONTRACT:
        raise SystemExit(f"index.json contract {index.get('contract')!r} != {CONTRACT!r}")
    cases, failures = [], []
    for case in index["cases"]:
        cid, ds = case["case_id"], case["dataset"]
        with h5py.File(truth_dir / "cases" / f"{cid}.h5", "r") as f:
            target = f["target"][()].astype(np.float64)
            fields = [str(x) for x in f.attrs["fields"]]
        if not np.isfinite(target).all():
            # staged from the pinned data by the verifier itself: a non-finite target is a data or
            # grader fault, never a measured model score, so it aborts scoring instead of becoming a 0
            raise ValueError(f"{cid}: the staged truth contains non-finite values")
        pred, reason = load_prediction(output_dir / "cases" / f"{cid}.h5", target.shape)
        if reason is not None:
            failures.append({"case_id": cid, "dataset": ds, "reason": reason})
            per = np.full((target.shape[0], target.shape[1]), VRMSE_MAX)
            nj = np.full(target.shape[0], VRMSE_MAX)
        else:
            per, nj = vrmse(pred, target), nrmse_joint(pred, target)
        by_step = per.mean(axis=1)
        cases.append({"case_id": cid, "dataset": ds, "failed": reason is not None,
                      "vrmse": float(by_step.mean()), "nrmse_joint": float(nj.mean()),
                      "vrmse_by_channel": {n: float(per[:, j].mean()) for j, n in enumerate(fields)},
                      "vrmse_by_step": [float(v) for v in by_step],
                      "windows": {w: float(by_step[a:b].mean()) for w, (a, b) in WINDOWS.items() if len(by_step) >= b}})
    datasets = {}
    for ds in sorted({c["dataset"] for c in cases}):
        cs = [c for c in cases if c["dataset"] == ds]
        d = {"n_cases": len(cs), "n_failed": sum(c["failed"] for c in cs),
             "n_steps": len(cs[0]["vrmse_by_step"]),
             "vrmse": float(np.mean([c["vrmse"] for c in cs])),
             "nrmse_joint": float(np.mean([c["nrmse_joint"] for c in cs]))}
        for w in WINDOWS:
            vals = [c["windows"][w] for c in cs if w in c["windows"]]
            if len(vals) == len(cs):
                d[f"vrmse_{w}"] = float(np.mean(vals))
        d["reward"] = reward_from_vrmse(d["vrmse"])
        datasets[ds] = d
    groups = {}
    for ds, d in datasets.items():
        g = "ape2d" if ds.startswith("ape2d/") else ds
        groups.setdefault(g, []).append(d["vrmse"])
    groups = {g: {"n_datasets": len(v), "vrmse": float(np.mean(v)), "reward": reward_from_vrmse(float(np.mean(v)))}
              for g, v in groups.items()}
    mean_vrmse = float(np.mean([g["vrmse"] for g in groups.values()])) if groups else VRMSE_MAX
    reward = float(np.mean([g["reward"] for g in groups.values()])) if groups else 0.0
    return {"contract": CONTRACT, "n_cases": len(cases), "n_failed": len(failures),
            "failures": failures, "datasets": datasets, "groups": groups, "mean_vrmse": mean_vrmse,
            "reward": reward, "cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--reward-json", help="also write {'reward': x} here")
    args = ap.parse_args()
    report = score(Path(args.input), Path(args.truth), Path(args.output))
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    if args.reward_json:
        Path(args.reward_json).write_text(json.dumps({"reward": report["reward"]}) + "\n")
    print(json.dumps({k: report[k] for k in ("n_cases", "n_failed", "mean_vrmse", "reward")}), file=sys.stderr)
    for ds, d in report["datasets"].items():
        print(f"  {ds:40s} cases={d['n_cases']:4d} failed={d['n_failed']:3d} K={d['n_steps']:3d} VRMSE={d['vrmse']:.4f}", file=sys.stderr)
    for g, d in report["groups"].items():
        print(f"  group {g:34s} datasets={d['n_datasets']:2d} VRMSE={d['vrmse']:.4f} reward={d['reward']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()

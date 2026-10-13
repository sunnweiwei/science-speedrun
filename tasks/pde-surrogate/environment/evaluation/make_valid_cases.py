#!/usr/bin/env python3
"""Build a validation case list from the manifest's `valid` pools, for self-service evaluation.

    python3 make_valid_cases.py --manifest split.json --out valid_cases.json [--per-trajectory 1] [--seed 0]
Each case starts at a deterministic offset so that start + 4 + K <= trajectory length,
with K taken from the manifest's per-source n_steps_out. Feed the result to make_cases.py.
"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per-trajectory", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="", help="if given, read true trajectory lengths from the files; else use the lock shapes")
a = ap.parse_args()
man = json.loads(Path(a.manifest).read_text())
lock = json.loads((Path(a.manifest).parent / "sources.lock.json").read_text())
shapes = {(s["id"], f["path"]): f["trajectory_shape_tchw"][0] for s in lock["sources"] for f in s["files"] if f["kind"] == "hdf5"}
K = man["n_steps_out"]; ctx = man["context_frames"]; rng = random.Random(a.seed)
cases = []
for p in man["pools"]["valid"]:
    fam = "ape2d" if p["source"].startswith("ape2d") else ("pdearena" if p["source"].startswith("pdearena") else "well")
    T, k = shapes[(p["source"], p["file"])], K[fam]
    for t in range(*p["trajectories"]):
        for j in range(a.per_trajectory):
            start = rng.randrange(0, T - ctx - k + 1)
            cases.append({"case_id": f"valid-{p['dataset'].replace('/', '-')}-{Path(p['file']).stem}-{t}-{j}",
                          "dataset": p["dataset"], "source": p["source"], "file": p["file"], "trajectory": t,
                          "start": start, "n_steps_out": k, "split": "valid"})
Path(a.out).write_text(json.dumps({"contract": man["contract"], "cases": cases}, indent=0) + "\n")
print(f"{len(cases)} validation cases -> {a.out}")

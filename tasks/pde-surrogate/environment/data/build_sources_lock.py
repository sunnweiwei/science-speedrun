#!/usr/bin/env python3
"""Regenerate sources.lock.json from the Hugging Face API at pinned revisions.

The lock records, for every HDF5 file of every pinned source: the canonical
path, byte size, upstream LFS SHA-256, split, and the trajectory count and
array shape inferred from the byte size. Trajectory counts are inferred, not
read, so they are cross-checked against the exact float32 payload size and the
file is rejected if the size is not consistent with an integer count.

Run with network access:  python3 build_sources_lock.py [--out sources.lock.json]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

SOURCES = [
    {
        "id": "ape2d",
        "repo": "thuerey-group/pde-transformer-ape2d",
        "revision": "abd0a02dab9ee28dda6cc7ef5cbaa95ce414c942",
        "license": "MIT",
        "family": "ape2d",
        "role": "pretraining corpus, reduced release (60 trajectories per PDE, 10 per Gray-Scott variant); strict prefix of ape2d-full",
    },
    {
        "id": "ape2d-full",
        "repo": "thuerey-group/pde-transformer-ape2d-full",
        "revision": "fa711d56a1a58705c3d8e889830b459fd92733f0",
        "license": "MIT",
        "family": "ape2d",
        "role": "pretraining corpus, full release (600 trajectories per PDE, 100 per Gray-Scott variant); the paper's training set",
    },
    {"id": "pdearena-ns", "repo": "pdearena/NavierStokes-2D",
     "revision": "cd99556a883a20acb9102c1f4bfdfae66ae33495", "license": "MIT", "family": "pdearena",
     "role": "PDEArena incompressible Navier-Stokes 2D (buoyant smoke, PhiFlow), 128^2, 14 stored steps; one HDF5 shard per seed and split"},
    {"id": "well-active_matter", "repo": "polymathic-ai/active_matter",
     "revision": "dc3e9135a75b4e5a0d979086219fc97fe668f7a3", "license": "CC-BY-4.0", "family": "well",
     "role": "downstream / transfer dataset (The Well)"},
    {"id": "well-turbulent_radiative_layer_2D", "repo": "polymathic-ai/turbulent_radiative_layer_2D",
     "revision": "2ee7756575ff6f90981d0308cbcb6a2ab5995bdc", "license": "CC-BY-4.0", "family": "well",
     "role": "downstream / transfer dataset (The Well)"},
    {"id": "well-rayleigh_benard", "repo": "polymathic-ai/rayleigh_benard",
     "revision": "10e143ebedae8a9b1c699b95d5b1eb8feaae09b5", "license": "CC-BY-4.0", "family": "well",
     "role": "downstream / transfer dataset (The Well); very large, only selected files are materialized"},
    {"id": "well-shear_flow", "repo": "polymathic-ai/shear_flow",
     "revision": "fc867f856f306905cf94c1f5df978cc518a2048c", "license": "CC-BY-4.0", "family": "well",
     "role": "downstream / transfer dataset (The Well); very large, only selected files are materialized"},
]

# ape2d array layout: sims/simN -> (T, C, 256, 256) float32
APE_CHANNELS = {"burgers": 2, "kdv": 2}
APE_TEST_STEPS = {"ks_test": 200, "decay_turb_test": 200, "kolm_flow_test": 200}

# The Well layout: t0_fields/<f> (n,T,X,Y), t1_fields/<f> (n,T,X,Y,2), t2_fields/<f> (n,T,X,Y,2,2)
WELL_SHAPES = {  # (T, C_total, X, Y)
    "active_matter": (81, 11, 256, 256),
    "turbulent_radiative_layer_2D": (101, 4, 128, 384),
    "rayleigh_benard": (200, 4, 512, 128),
    "shear_flow": (200, 4, 256, 512),
}
FLOAT_BYTES = 4
# PDEArena NS shards: root group <split> with u, vx, vy of shape (n, 14, 128, 128) float64
PDEARENA_SHAPE = (14, 3, 128, 128)


def fetch(repo, revision):
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def ape_shape(stem):
    base = stem[:-5] if stem.endswith("_test") else stem
    channels = 2 if base.startswith("gs_") else APE_CHANNELS.get(base, 1)
    if stem.endswith("_test"):
        steps = APE_TEST_STEPS.get(stem, 100)  # gs_*_test files hold 100 steps
    else:
        steps = 30
    return steps, channels, 256, 256


def infer_count(size, shape, family):
    """Infer the trajectory count from the byte size.

    ape2d files are contiguous float32 payload plus ~1 MB of normalization
    buffers, so the size must match an integer count within 1% + 16 MB.
    The Well files carry unallocated HDF5 space of up to ~40% of one
    trajectory's payload (measured on downloaded files), so the count is the
    floor of size / payload and the remainder must stay below one payload.
    Both are inferences; materialize.py reads the true count from each file.
    """
    per = FLOAT_BYTES
    for d in shape:
        per *= d
    if family == "pdearena":
        per *= 2  # float64
        n = round(size / per)
        ok = n >= 1 and abs(size - n * per) <= 0.01 * size + 16 * 2**20
    elif family == "ape2d":
        n = round(size / per)
        ok = n >= 1 and abs(size - n * per) <= 0.01 * size + 16 * 2**20
    else:
        n = size // per
        ok = n >= 1 and 0 <= size - n * per < per
    return n if ok else None


def canonical_ape(path):
    """The reduced release stores every file up to three times; keep '<stem>/<stem>.hdf5'."""
    parts = path.split("/")
    stem = parts[-1][:-5]
    return len(parts) == 2 and parts[0] == stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("sources.lock.json")))
    args = ap.parse_args()
    lock = {"schema": "pde-surrogate-sources-lock/v1", "sources": []}
    problems = []
    for src in SOURCES:
        info = fetch(src["repo"], src["revision"])
        assert info["sha"] == src["revision"], (src["repo"], info["sha"])
        files, seen = [], {}
        for sib in sorted(info["siblings"], key=lambda s: s["rfilename"]):
            path, size = sib["rfilename"], sib.get("size") or 0
            lfs = sib.get("lfs") or {}
            if not path.endswith((".hdf5", ".h5")):
                if not path.startswith((".git", "gif/", "visualization")):
                    files.append({"path": path, "bytes": size, "sha256": lfs.get("sha256"), "kind": "metadata"})
                continue
            sha = lfs.get("sha256")
            if src["family"] == "ape2d":
                if src["id"] == "ape2d" and not canonical_ape(path):
                    seen.setdefault(sha, []).append(path)
                    continue
                stem = path.split("/")[-1][:-5]
                shape = ape_shape(stem)
                split = "test" if stem.endswith("_test") else "train"
                name = stem
            elif src["family"] == "pdearena":
                shape = PDEARENA_SHAPE
                split = path.split("_")[1]
                name = path[:-3]
            else:
                ds = src["repo"].split("/")[-1]
                shape = WELL_SHAPES[ds]
                split = path.split("/")[1]
                name = path.split("/")[-1][:-5]
            n = infer_count(size, shape, src["family"])
            if n is None:
                problems.append(f"{src['repo']}:{path} size {size} inconsistent with shape {shape}")
            files.append({"path": path, "bytes": size, "sha256": sha, "kind": "hdf5", "name": name,
                          "dtype": "float64" if src["family"] == "pdearena" else "float32",
                          "split": split, "n_trajectories_inferred": n,
                          "trajectory_shape_tchw": list(shape)})
        entry = {**src, "hf_sha": info["sha"], "hf_last_modified": info.get("lastModified"), "files": files}
        if seen:
            entry["duplicate_paths_dropped"] = seen
        h5 = [f for f in files if f["kind"] == "hdf5"]
        entry["summary"] = {
            "hdf5_files": len(h5),
            "hdf5_bytes": sum(f["bytes"] for f in h5),
            "trajectories_by_split": {
                s: sum(f["n_trajectories_inferred"] or 0 for f in h5 if f["split"] == s)
                for s in sorted({f["split"] for f in h5})},
        }
        lock["sources"].append(entry)
        print(f"{src['id']:36s} {entry['summary']}", file=sys.stderr)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    Path(args.out).write_text(json.dumps(lock, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

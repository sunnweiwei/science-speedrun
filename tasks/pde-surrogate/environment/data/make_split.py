#!/usr/bin/env python3
"""Generate the task's trajectory-level split manifest from sources.lock.json.

    python3 make_split.py --corpus full    -> split.json        (canonical; ape2d-full + Well + PDEArena)
    python3 make_split.py --corpus reduced -> split_smoke.json  (plumbing split on the reduced release; ape2d + Well + PDEArena)

Protocol (all sources): 4 context frames, rollout from frame 0, one held-out
case per test trajectory, K predicted steps per source:
    ape2d 26 (30 stored frames), The Well 30, PDEArena 10 (14 stored frames).
The long-rollout ape2d test files additionally get K=96 cases in split
"test_long" (secondary evaluation, opt-in).

Pretraining corpus indices follow the paper's loader (ape_2d_xxl.py) with a
validation block carved out of its training range; The Well and PDEArena keep
their upstream split directories.
"""
import argparse
import json
from pathlib import Path

LOCK = Path(__file__).with_name("sources.lock.json")
CONTEXT = 4
K = {"ape2d": 26, "well": 30, "pdearena": 10}
K_LONG = 96
# PDEs whose held-out set is a separate *_test file (paper loader)
SEPARATE_TEST = {"ks", "decay_turb", "kolm_flow", "gs_alpha", "gs_beta", "gs_gamma", "gs_epsilon"}
IN_FILE_SPLIT = {"diff", "burgers", "kdv", "fisher", "sh", "gs_delta", "gs_theta", "gs_iota", "gs_kappa"}
# `hyp.hdf5` in the full release (thuerey-group/pde-transformer-ape2d-full @ fa711d56) is not
# hyper-diffusion: its `sims` group is a byte-level copy of `diff.hdf5` (same `PDE = "Diffusion"`
# attribute, same constants, identical per-trajectory arrays and an identical sidecar json), so a
# split that used it would score one family twice and double-weight diffusion in the corpus group.
# It is excluded from every pool and from the held-out set; the reduced release's `hyp` is a real
# hyper-diffusion set, but the same family list is used for both corpora so the smoke split
# mirrors the canonical one. Reported upstream; see docs/AUDIT-decisions.md.
EXCLUDED_UPSTREAM_DUPLICATES = {"hyp"}
PDEARENA_TEST_SHARDS = 8   # 8 x 25 = 200 held-out trajectories, deterministic by sorted shard name
PDEARENA_VALID_SHARDS = 8


def ape_ranges(n, name, corpus):
    """(train, valid, test) index ranges for an ape2d training file with n simulations."""
    if name in SEPARATE_TEST:
        v = max(1, n // 10)
        return (0, n - v), (n - v, n), None
    if name.startswith("gs_"):        # 100 sims full / 10 reduced, paper: 80/20
        t0 = int(n * 0.8)
        v = max(1, int(n * 0.08))
        return (0, t0 - v), (t0 - v, t0), (t0, n)
    t0 = int(n * 5 / 6)               # 600 -> 500 / 60 -> 50, paper: 500/100 and 50/10
    v = max(1, n // 30)               # 20 of 600, 2 of 60
    return (0, t0 - v), (t0 - v, t0), (t0, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["full", "reduced"], default="full")
    ap.add_argument("--out")
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    by_id = {s["id"]: s for s in lock["sources"]}
    ape_id = "ape2d-full" if args.corpus == "full" else "ape2d"
    pools = {"train": [], "valid": []}
    cases = []

    def add_case(source, dataset, file, traj, k, split, cid):
        cases.append({"case_id": cid, "dataset": dataset, "source": source, "file": file,
                      "trajectory": traj, "start": 0, "n_steps_out": k, "split": split})

    # --- pretraining corpus
    src = by_id[ape_id]
    files = {f["name"]: f for f in src["files"] if f["kind"] == "hdf5"}
    for name in sorted(IN_FILE_SPLIT | SEPARATE_TEST):
        f = files[name]
        n = f["n_trajectories_inferred"]
        tr, va, te = ape_ranges(n, name, args.corpus)
        pools["train"].append({"source": ape_id, "dataset": f"ape2d/{name}", "file": f["path"], "trajectories": list(tr)})
        pools["valid"].append({"source": ape_id, "dataset": f"ape2d/{name}", "file": f["path"], "trajectories": list(va)})
        if te is not None:
            for i in range(*te):
                add_case(ape_id, f"ape2d/{name}", f["path"], i, K["ape2d"], "test", f"{ape_id}-{name}-{i:04d}")
        else:
            tf = files[f"{name}_test"]
            for i in range(tf["n_trajectories_inferred"]):
                add_case(ape_id, f"ape2d/{name}", tf["path"], i, K["ape2d"], "test", f"{ape_id}-{name}-test-{i:04d}")
                add_case(ape_id, f"ape2d/{name}", tf["path"], i, K_LONG, "test_long", f"{ape_id}-{name}-testlong-{i:04d}")
    # --- The Well (upstream split directories, every trajectory)
    for sid in ("well-turbulent_radiative_layer_2D", "well-active_matter"):
        src = by_id[sid]
        ds = "well/" + src["repo"].split("/")[-1]
        for f in sorted((x for x in src["files"] if x["kind"] == "hdf5"), key=lambda x: x["path"]):
            n = f["n_trajectories_inferred"]
            if f["split"] in ("train", "valid"):
                pools[f["split"]].append({"source": sid, "dataset": ds, "file": f["path"], "trajectories": [0, n]})
            else:
                for i in range(n):
                    add_case(sid, ds, f["path"], i, K["well"], "test", f"{sid}-{f['name']}-{i}")
    # --- PDEArena Navier-Stokes
    src = by_id["pdearena-ns"]
    shards = {s: sorted((x for x in src["files"] if x["kind"] == "hdf5" and x["split"] == s), key=lambda x: x["path"]) for s in ("train", "valid", "test")}
    for f in shards["train"]:
        pools["train"].append({"source": "pdearena-ns", "dataset": "pdearena/navier_stokes_2d", "file": f["path"], "trajectories": [0, f["n_trajectories_inferred"]]})
    for f in shards["valid"][:PDEARENA_VALID_SHARDS]:
        pools["valid"].append({"source": "pdearena-ns", "dataset": "pdearena/navier_stokes_2d", "file": f["path"], "trajectories": [0, f["n_trajectories_inferred"]]})
    for f in shards["test"][:PDEARENA_TEST_SHARDS]:
        seed = f["name"].split("_")[2]
        for i in range(f["n_trajectories_inferred"]):
            add_case("pdearena-ns", "pdearena/navier_stokes_2d", f["path"], i, K["pdearena"], "test", f"pdearena-{seed}-{i:02d}")

    out = {"schema": "pde-surrogate-split/v1", "contract": "pde-surrogate-rollout/v1", "corpus": args.corpus,
           **({"warning": "plumbing only: the reduced release is a different draw of the corpus parameters, not a subset of "
                          "the full release; never compare smoke numbers with canonical ones"} if args.corpus == "reduced" else {}),
           "context_frames": CONTEXT, "n_steps_out": K, "n_steps_out_long": K_LONG, "pools": pools, "cases": cases}
    path = Path(args.out or LOCK.with_name("split.json" if args.corpus == "full" else "split_smoke.json"))
    path.write_text(json.dumps(out, indent=0) + "\n")
    import collections
    per = collections.Counter((c["dataset"].split("/")[0], c["split"]) for c in cases)
    ntr = sum(p["trajectories"][1] - p["trajectories"][0] for p in pools["train"])
    nva = sum(p["trajectories"][1] - p["trajectories"][0] for p in pools["valid"])
    print(f"{path.name}: train {ntr} traj, valid {nva} traj, cases {dict(per)}")


if __name__ == "__main__":
    main()

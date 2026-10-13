#!/usr/bin/env python3
"""Prove that the public training release contains no held-out trajectory.

    python3 check_release_leakage.py --release RELEASE_DIR --data $TASK_PRIVATE_DIR \
        --out release_leakage.json

The release is built by omitting every test-only file and truncating the ten corpus files
that carry both training and held-out trajectories. That is an argument about indices; this
script checks it against content, which is the only form that survives an off-by-one.

For every held-out case in split.json it hashes the trajectory array from the *source*
materialization, then hashes every trajectory present in the release, and asserts the two
sets are disjoint. It also re-reads the release's own structure: a truncated file must hold
exactly `keep` trajectories with no gaps, and a test-only file must be absent entirely.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[2] / "environment" / "data"


def traj_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="release_leakage.json")
    ap.add_argument("--limit-files", type=int, default=0, help="only check the first N release files")
    a = ap.parse_args()

    split = json.loads((HERE / "split.json").read_text())
    release, data = Path(a.release), Path(a.data)
    manifest = json.loads((release / "release-manifest.json").read_text())
    records = [json.loads(l) for l in (release / "payload-files.jsonl").read_text().splitlines() if l.strip()]
    # paths are "<subtree>/<source>/<file>"; only train/ is ever mounted for an agent, so that is
    # the subtree this check is about. eval/ is expected to hold held-out data - that is its job.
    train = {r["path"].split("/", 1)[1]: r for r in records if r.get("subtree", "train") == "train"}
    present = set(train)
    derived_only = manifest.get("kind") == "derived-only"

    held = defaultdict(set)
    for c in split["cases"]:
        if "file" in c:
            held[(c["source"], c["file"])].add(int(c["trajectory"]))
    pools = defaultdict(list)
    for name in ("train", "valid"):
        for e in split["pools"][name]:
            pools[(e["source"], e["file"])].append(tuple(e["trajectories"]))

    checked_files = checked_traj = 0
    structure, collisions, missing = [], [], []
    test_only_present = []

    for key, trajs in sorted(held.items()):
        rel_name = f"{key[0]}/{key[1]}"
        if key not in pools:
            if rel_name in present:                     # a held-out-only file must not be in train/
                test_only_present.append(rel_name)
            continue
        if rel_name not in present:
            # a derived-only release carries just the files this split rewrites; the rest come from
            # upstream untouched and have nothing to check
            if not derived_only:
                missing.append(rel_name)
            continue
        rp = release / train[rel_name]["path"]
        if a.limit_files and checked_files >= a.limit_files:
            break
        keep = max(hi for _, hi in pools[key])
        src = data / "hf" / key[0] / key[1]
        if not src.exists():
            src = data / key[0] / key[1]

        with h5py.File(rp, "r") as fr:
            have = sorted(int(n[3:]) for n in fr["sims"] if n.startswith("sim"))
            structure.append({"file": rel_name, "expected_keep": keep, "found": len(have),
                              "contiguous_from_zero": have == list(range(len(have))),
                              "ok": have == list(range(keep))})
            rel_hashes = {traj_hash(fr["sims"][f"sim{i}"][()]) for i in have}
        with h5py.File(src, "r") as fs:
            for t in sorted(trajs):
                h = traj_hash(fs["sims"][f"sim{t}"][()])
                checked_traj += 1
                if h in rel_hashes:
                    collisions.append({"file": rel_name, "held_out_trajectory": t})
        checked_files += 1
        print(f"  {rel_name:44s} kept {len(have):4d}/{keep:4d}   held-out checked {len(trajs):4d}   "
              f"{'CLEAN' if not collisions else 'LEAK'}")

    clean = not collisions and not test_only_present and not missing and all(s["ok"] for s in structure)
    doc = {
        "what": "content check that the release's train/ subtree carries no held-out trajectory",
        "release_kind": manifest.get("kind", "mirror"),
        "method": "hash every trajectory array present in the release, hash every held-out trajectory from the "
                  "source materialization, and require the two sets to be disjoint. Structure is re-read from the "
                  "release rather than assumed from the build.",
        "release_id": manifest["release_id"], "release_files": manifest["payload_files"],
        "files_checked": checked_files, "held_out_trajectories_checked": checked_traj,
        "truncated_file_structure": structure,
        "content_collisions": collisions,
        "test_only_files_wrongly_present": test_only_present,
        "expected_files_missing": missing,
        "clean": clean,
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\n{checked_traj} held-out trajectories checked against {checked_files} release files")
    print("RESULT:", "clean" if clean else "LEAK DETECTED")
    raise SystemExit(0 if clean else 1)


if __name__ == "__main__":
    main()

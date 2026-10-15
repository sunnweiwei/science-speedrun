#!/usr/bin/env python3
"""Split the built evaluation suite into the three views of the release.

  OUT/public/downstream/<task>/{train,valid}.lmdb   agent + verifier: fine-tune data with labels
  OUT/public/downstream/task_metainfo.json          train-split mean/std for target normalisation (non-official tasks)
  OUT/public/downstream/manifest.json               per task: group, targets, sizes
  OUT/verifier/test-inputs/<task>/test.lmdb         verifier main only: test molecules, labels replaced by 0.0
  OUT/private/labels/<task>.csv                     grader only: one row per test SMILES, target_0..target_{k-1}

Test labels are grouped by SMILES (mean over duplicates), the aggregation of Uni-Mol2's agg_mae; predictions are
grouped the same way (evaluation/protocol/predict.py).
usage: make_views.py SUITE UNI2_REPORT OUT   (SUITE: output of build_fast.py full-test + build_admet.py)
"""
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import lmdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "protocol"))
from tasks import GROUP_OF, NUM_TARGETS, TASKS  # noqa: E402


def read(path: Path) -> list[dict]:
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        rows = [pickle.loads(txn.get(str(i).encode())) for i in range(txn.stat()["entries"])]
    env.close()
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=1 << 36, lock=False)
    with env.begin(write=True) as txn:
        for i, r in enumerate(rows):
            txn.put(str(i).encode(), pickle.dumps(r, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync(); env.close()


def main(suite: Path, uni2_report: Path, out: Path) -> None:
    pub, ver, priv = out / "public" / "downstream", out / "verifier" / "test-inputs", out / "private" / "labels"
    report = json.loads(uni2_report.read_text())
    meta = {t: {"mean": report[t]["train_mean"][0], "std": report[t]["train_std"][0]}
            for t in TASKS if t.startswith("compas1d_")}
    meta.update(json.loads((suite / "task_metainfo.json").read_text()))
    manifest = {"tasks": {}}
    for task in TASKS:
        k = NUM_TARGETS[task]
        for split in ("train", "valid"):
            (pub / task).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(suite / task / f"{split}.lmdb", pub / task / f"{split}.lmdb")
        test = read(suite / task / "test.lmdb")
        placeholder = tuple([0.0] * k) if k > 1 else 0.0
        write(ver / task / "test.lmdb", [{"atoms": r["atoms"], "coordinates": r["coordinates"], "smi": r["smi"],
                                          "target": placeholder} for r in test])
        labels: dict[str, list[list[float]]] = {}
        for r in test:
            y = list(r["target"]) if k > 1 else [r["target"]]
            labels.setdefault(r["smi"], []).append([float(v) for v in y])
        priv.mkdir(parents=True, exist_ok=True)
        with open(priv / f"{task}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["smi"] + [f"target_{j}" for j in range(k)])
            for smi in sorted(labels):
                ys = labels[smi]
                w.writerow([smi] + [repr(sum(y[j] for y in ys) / len(ys)) for j in range(k)])
        manifest["tasks"][task] = {"group": GROUP_OF[task], "targets": k,
                                   "train": len(read(pub / task / "train.lmdb")),
                                   "valid": len(read(pub / task / "valid.lmdb")),
                                   "test_molecules": len(test), "test_smiles": len(labels)}
        print(task, manifest["tasks"][task], flush=True)
    missing = [t for t in TASKS if t not in meta and not t.startswith("qm9")]
    assert not missing, missing
    (pub / "task_metainfo.json").write_text(json.dumps(meta, indent=1) + "\n")
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

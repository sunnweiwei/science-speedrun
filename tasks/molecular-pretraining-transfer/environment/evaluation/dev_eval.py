#!/usr/bin/env python3
"""Development evaluation: the verifier's fine-tune protocol (protocol/) on the development data, scored on the
validation split.

    python /workspace/evaluation/dev_eval.py --encoder PKG|none --out /workspace/research/eval/NAME
                                             [--tasks T ...] [--gpus 0,1,...] [--slots 3] [--epochs 15]

The verifier fine-tunes on train, keeps the best-validation epoch and predicts the held-out test split, which is
not in this container. Here the validation split plays the test split's role (it is also used for epoch
selection, so the numbers are slightly optimistic). Same metrics as the verifier: per task MAE, RAE, R2, Spearman,
Kendall; per group MA-RAE and mean Spearman. Writes OUT/report.json and prints the group table. On 8 H100 the full
24-task suite takes about 30 minutes for one encoder.
"""
import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "protocol"))
from tasks import GROUPS, NUM_TARGETS, TASKS, group_summary, task_metrics  # noqa: E402

DATA = Path("/workspace/data/downstream")


def labels(path: Path, k: int) -> dict[str, list[float]]:
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    acc: dict[str, list[list[float]]] = {}
    with env.begin() as txn:
        for _, v in txn.cursor():
            r = pickle.loads(v)
            acc.setdefault(r["smi"], []).append(list(r["target"]) if k > 1 else [r["target"]])
    return {s: [sum(y[j] for y in ys) / len(ys) for j in range(k)] for s, ys in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--gpus")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15, help="the verifier always uses 15")
    a = ap.parse_args()
    view = a.out / "data"
    for t in a.tasks:
        (view / t).mkdir(parents=True, exist_ok=True)
        for split, src in (("train", "train"), ("valid", "valid"), ("test", "valid")):
            link = view / t / f"{split}.lmdb"
            if not link.exists():
                link.symlink_to(DATA / t / f"{src}.lmdb")
    if not (view / "task_metainfo.json").exists():
        (view / "task_metainfo.json").symlink_to(DATA / "task_metainfo.json")
    cmd = [sys.executable, str(HERE / "protocol" / "run_suite.py"), "--encoder", a.encoder, "--data", str(view),
           "--out", str(a.out / "runs"), "--tasks", *a.tasks, "--slots", str(a.slots)] + (["--gpus", a.gpus] if a.gpus else [])
    subprocess.run(cmd, check=True, env={**__import__("os").environ, "FT_EPOCHS": str(a.epochs)})
    per_task = {}
    for t in a.tasks:
        k = NUM_TARGETS[t]
        with open(a.out / "runs" / t / "predictions.csv", newline="") as f:
            rows = list(csv.reader(f))[1:]
        per_task[t] = task_metrics({r[0]: [float(v) for v in r[1:]] for r in rows}, labels(DATA / t / "valid.lmdb", k))
    report = {"encoder": a.encoder, "split": "valid", "epochs": a.epochs, "tasks": per_task}
    if set(a.tasks) == set(TASKS):
        report["groups"] = group_summary(per_task)
    (a.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'task':36s}{'MAE':>10s}{'RAE':>8s}{'Spearman':>10s}")
    for g, ts in GROUPS.items():
        for t in (t for t in ts if t in per_task):
            m = per_task[t]
            print(f"{t:36s}{m['mae']:10.4g}{m['rae']:8.3f}{m['spearman']:10.3f}")
    for k, v in report.get("groups", {}).items():
        print(f"{k:36s}{v:10.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trusted grader: score the sealed test predictions against the private labels. Never runs package code.

Waits for the main verifier's READY (or INVALID) marker, checks the snapshot hashes, and writes
OUT/reward.json (flat numbers: valid, per-group MA-RAE and mean Spearman, per-task MAE / RAE / R2 / Spearman /
Kendall) and OUT/report.json (the same with encoder check and fine-tune timings). Raw metrics, no aggregate
(TASK_STANDARD.md multi-metric exception).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "protocol"))
from tasks import GROUPS, NUM_TARGETS, TASKS, group_summary, task_metrics  # noqa: E402


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path, prefix: str, k: int) -> dict[str, list[float]]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["smi"] + [f"{prefix}_{j}" for j in range(k)]:
        raise ValueError(f"{path.name}: bad header")
    out = {}
    for r in rows[1:]:
        if r[0] in out:
            raise ValueError(f"{path.name}: duplicate SMILES")
        out[r[0]] = [float(v) for v in r[1:]]
    return out


def validate_snapshot(sealed: Path) -> dict:
    if (sealed.parent / "READY").read_text().strip() != sha256(sealed / "snapshot.json"):
        raise ValueError("READY does not match the snapshot")
    snap = json.loads((sealed / "snapshot.json").read_text())
    if snap.get("schema") != "mpt-v2-predictions/v1":
        raise ValueError("unknown snapshot schema")
    present = {p.name for p in sealed.iterdir()} - {"snapshot.json"}
    if present != set(snap["files"]) or present != {f"{t}.csv" for t in TASKS} | {"encoder_check.json", "suite.json"}:
        raise ValueError("sealed file set differs from the snapshot")
    for name, digest in snap["files"].items():
        if sha256(sealed / name) != digest:
            raise ValueError(f"hash mismatch: {name}")
    return snap


def score(sealed: Path, labels: Path) -> tuple[dict, dict]:
    per_task = {}
    for t in TASKS:
        k = NUM_TARGETS[t]
        truth = load_csv(labels / f"{t}.csv", "target", k)
        pred = load_csv(sealed / f"{t}.csv", "predict", k)
        if set(pred) != set(truth):
            raise ValueError(f"{t}: predictions cover {len(set(pred) & set(truth))} of {len(truth)} test molecules")
        per_task[t] = task_metrics(pred, truth)
    groups = group_summary(per_task)
    reward = {"valid": 1.0, **groups}
    for t, m in per_task.items():
        for name in ("mae", "rae", "r2", "spearman", "kendall"):
            reward[f"{t}_{name}"] = m[name]
    report = {"groups": groups, "tasks": per_task, "group_tasks": GROUPS,
              "encoder_check": json.loads((sealed / "encoder_check.json").read_text()),
              "fine_tunes": json.loads((sealed / "suite.json").read_text())}
    return reward, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", type=Path, default=Path("/sealed/run"))
    ap.add_argument("--labels", type=Path, default=Path("/labels"))
    ap.add_argument("--output", type=Path, default=Path("/sealed/results"))
    ap.add_argument("--wait-seconds", type=int, default=6 * 3600)
    a = ap.parse_args()
    deadline = time.monotonic() + a.wait_seconds
    while not (a.sealed.parent / "READY").is_file():
        if (a.sealed.parent / "INVALID").is_file():
            print("main verifier rejected the submission; nothing to score")
            return
        if time.monotonic() > deadline:
            raise TimeoutError("no sealed predictions")
        time.sleep(2)
    validate_snapshot(a.sealed)
    reward, report = score(a.sealed, a.labels)
    tmp = a.output.with_name(a.output.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (tmp / "reward.json").write_text(json.dumps(harbor_reward(reward), indent=1) + "\n")
    tmp.rename(a.output)
    print(json.dumps({k: v for k, v in reward.items() if "_ma_rae" in k or k == "valid"}))


if __name__ == "__main__":
    main()

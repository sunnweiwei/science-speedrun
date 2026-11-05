#!/usr/bin/env python3
"""Summarize a longer local pilot against the one-target wiring smoke."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    root = args.run_root
    summary = read_json(root / "model/training_summary.json")
    scores = read_json(root / "predictions/metrics.json")
    epochs = [json.loads(line) for line in
              (root / "finetune-work/metrics.jsonl").read_text().splitlines()]
    assert summary["formal_recipe"] is False
    assert summary["completed_epochs"] == {"profsa": 3, "finetune": 6}
    assert len(epochs) == 6
    assert all(math.isfinite(row["train_loss"]) and
               math.isfinite(row["valid_bedroc"]) for row in epochs)
    best = max(epochs, key=lambda row: row["valid_bedroc"])
    assert summary["selected_epoch"] == best["epoch"]
    assert summary["best_valid_bedroc"] == best["valid_bedroc"]
    assert scores["benchmarks"]["dude"]["targets"] == 1
    target = scores["per_target"][0]
    assert target["target_id"] == "aa2ar" and target["ligands"] == 32010
    checkpoint = root / "standalone-model/model.pt"
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    begin = datetime.fromisoformat((root / "started_utc.txt").read_text().strip())
    end = datetime.fromisoformat((root / "finished_utc.txt").read_text().strip())
    result = {
        "schema": "drugclip-single-gpu-pilot/v1",
        "formal_recipe": False,
        "parameter_count": summary["parameter_count"],
        "profsa_records": summary["profsa"]["samples"],
        "profsa_epochs": summary["completed_epochs"]["profsa"],
        "finetune_records": summary["finetune"]["samples"],
        "finetune_epochs": summary["completed_epochs"]["finetune"],
        "dev_records": summary["finetune"]["dev_samples"],
        "per_gpu_batch": summary["per_gpu_batch"],
        "duration_seconds": int((end - begin).total_seconds()),
        "selected_epoch": best["epoch"],
        "best_dev_bedroc": best["valid_bedroc"],
        "dev_by_epoch": [{"epoch": row["epoch"], "train_loss": row["train_loss"],
                          "valid_loss": row["valid_loss"],
                          "valid_bedroc": row["valid_bedroc"],
                          "valid_auc": row["valid_auc"]} for row in epochs],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "test_scope": "one complete DUD-E aa2ar target after selection",
        "test_metrics": target,
        "model_checkpoint_sha256": model_hash,
    }
    if args.baseline_report and args.baseline_report.is_file():
        baseline = read_json(args.baseline_report)["test_metrics"]
        assert baseline["target_id"] == target["target_id"]
        result["baseline_model"] = "two-epoch eight-record wiring smoke"
        result["test_delta_vs_baseline"] = {
            key: target[key] - baseline[key]
            for key in ("roc_auc", "bedroc_80.5", "ef_0.005", "ef_0.01",
                        "ef_0.02", "ef_0.05")
        }
    (root / "pilot_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

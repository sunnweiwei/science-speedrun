"""Check completed representative learning curves before allocating the array.

This is a pretraining sanity/resource gate, never a convergence or test-quality
claim. It uses only synthetic validation and the completed throughput pilots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .cache import atomic_json


GROUPS = ("d3_bX_center_3_5", "d5_bX_center_5_5")


def check_representatives(asset_root: Path, campaign: str) -> dict[str, Any]:
    records = {}
    for group in GROUPS:
        directory = asset_root / campaign / "models" / group / "pretrain"
        progress = json.loads((directory / "progress.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        pilot = json.loads((asset_root / "pilot" / group / "progress.json").read_text())
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        validations = [entry for entry in history if entry.get("complete")]
        best = min(validations, key=lambda entry: entry["loss"])
        improvement = 1 - best["loss"] / pilot["latest_validation"]["loss"]
        conditions = {
            "stage_stopped": progress["status"] == "stopped" and bool(metadata["sessions"][-1].get("ended_at")),
            "completed_at_least_one_epoch": progress["epoch"] >= 1,
            "validation_population_complete": all(entry["shots"] == 66560 for entry in validations),
            "finite_validation": all(math.isfinite(entry[key]) for entry in validations for key in ("loss", "main_loss", "aux_loss", "main_error")),
            "combined_validation_loss_improved_5pct_over_pilot": improvement >= 0.05,
            "selected_main_validation_error_below_46pct": best["main_error"] < 0.46,
            "checkpoint_exists": (directory / "best.keras").is_file(),
        }
        records[group] = {
            "passed": all(conditions.values()), "checks": conditions,
            "epochs_completed": progress["epoch"], "seen_samples": progress["seen_samples"],
            "elapsed_seconds": progress["elapsed_seconds"], "selected_validation": best,
            "relative_loss_improvement_over_pilot": improvement,
        }
    result = {
        "passed": all(record["passed"] for record in records.values()), "groups": records,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic validation learning and resource sanity; not convergence or experimental test performance",
    }
    result["status"] = "passed" if result["passed"] else (
        "failed" if all(record["checks"]["stage_stopped"] for record in records.values()) else "waiting_for_completed_stages")
    atomic_json(asset_root / campaign / "representative-gate.json", result)
    if not result["passed"]:
        raise RuntimeError("Representative training gate failed; inspect saved diagnostics before allocating the full array")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--campaign", default="native-v1")
    args = parser.parse_args()
    print(json.dumps(check_representatives(args.asset_root, args.campaign)))


if __name__ == "__main__":
    main()

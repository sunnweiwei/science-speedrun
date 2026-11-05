#!/usr/bin/env python3
"""Check training transitions and the complete-target smoke artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import sha256
from model import PaperConfig, PaperDrugCLIP, parameter_count
from train import MOLECULE_SHA256


def changed(left: dict, right: dict, prefix: str) -> bool:
    return any(not torch.equal(value, right[key])
               for key, value in left.items() if key.startswith(prefix))


def unchanged(left: dict, right: dict, prefix: str) -> bool:
    return not changed(left, right, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("smoke_root", type=Path)
    args = parser.parse_args()
    root = args.smoke_root
    profsa = torch.load(root / "profsa-work/profsa_complete.pt", map_location="cpu",
                        weights_only=False)
    last = torch.load(root / "finetune-work/last.pt", map_location="cpu",
                      weights_only=False)
    stopped = torch.load(root / "finetune_epoch1_last.pt", map_location="cpu",
                         weights_only=False)
    summary = json.loads((root / "standalone-model/training_summary.json").read_text())
    metrics = json.loads((root / "predictions/metrics.json").read_text())
    stage_log = [json.loads(line) for line in
                 (root / "finetune-work/metrics.jsonl").read_text().splitlines()]

    torch.manual_seed(1)
    initial = PaperDrugCLIP(PaperConfig())
    initial.molecule_encoder.load_official_checkpoint(
        args.data_root / "pretrained/mol_pre_no_h_220816.pt", MOLECULE_SHA256)
    original = initial.state_dict()
    pretrain = profsa["model"]
    final = last["model"]
    assert unchanged(original, pretrain, "molecule_encoder.")
    assert changed(original, pretrain, "pocket_encoder.")
    assert changed(original, pretrain, "molecule_projection.")
    assert changed(pretrain, final, "molecule_encoder.")
    assert changed(pretrain, final, "pocket_encoder.")
    assert parameter_count(initial) == summary["parameter_count"] == 95_305_418
    assert profsa["complete"] and profsa["epoch"] == 2
    assert last["stage"] == "finetune" and last["epoch"] == 2
    assert stopped["stage"] == "finetune" and stopped["epoch"] == 1
    assert stopped["stop_reason"] == "time_budget"
    assert stopped["completed_epochs"] == {"profsa": 2, "finetune": 1}
    assert stopped["scheduler"]["updates"] == 4
    assert stopped["completed_updates"] == {"profsa": 8, "finetune": 4}
    assert stopped["global_updates"] == 12
    assert stopped["optimizer"]["state"]
    assert stopped["scaler"] is not None and stopped["rng_by_rank"]
    assert stopped["best_epoch"] == 1
    assert stopped["best_bedroc"] == stage_log[0]["valid_bedroc"]
    assert last["stop_reason"] == summary["stop_reason"] == "max_epochs"
    assert last["completed_epochs"] == {"profsa": 2, "finetune": 2}
    assert last["scheduler"]["updates"] == last["completed_updates"]["finetune"]
    assert last["global_updates"] == sum(last["completed_updates"].values())
    assert last["optimizer"]["state"] and last["scaler"] is not None
    assert last["elapsed_seconds"] > stopped["elapsed_seconds"]
    assert last["best_bedroc"] == summary["best_valid_bedroc"]
    assert stage_log[0]["stop_decision"] == "time_budget"
    assert stage_log[1]["stop_decision"] == "max_epochs"
    assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in stage_log)
    assert all(torch.isfinite(torch.tensor(row["valid_bedroc"])) for row in stage_log)
    assert summary["formal_recipe"] is False
    assert summary["pocket_crop"]["policy"] == "author_distance_weighted_v1"
    assert summary["pocket_crop"]["seed"] == 1
    assert summary["selected_epoch"] == max(
        stage_log, key=lambda row: row["valid_bedroc"])["epoch"]
    assert summary["best_valid_bedroc"] == max(row["valid_bedroc"] for row in stage_log)
    assert metrics["benchmarks"]["dude"]["targets"] == 1
    assert len(metrics["per_target"]) == 1
    assert metrics["per_target"][0]["ligands"] > 1000
    result = {
        "schema": "drugclip-single-gpu-smoke/v1",
        "formal_recipe": False,
        "parameter_count": parameter_count(initial),
        "profsa_frozen_molecule": True,
        "profsa_updated_pocket_and_projection": True,
        "finetune_updated_both_backbones": True,
        "resume_optimizer_scheduler_amp_updates_best": True,
        "stop_then_resume": [row["stop_decision"] for row in stage_log],
        "dev": stage_log,
        "selected_epoch": summary["selected_epoch"],
        "selected_bedroc": summary["best_valid_bedroc"],
        "test_scope": "one complete DUD-E target after selection",
        "test_metrics": metrics["per_target"][0],
        "model_checkpoint_sha256": sha256(root / "standalone-model/model.pt"),
        "peak_cuda_allocated_bytes": max(
            row.get("peak_cuda_allocated_bytes", 0) for row in stage_log),
    }
    (root / "smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

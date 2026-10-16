"""Audit pinned project configs and emit reviewable Large fine-tuning candidates."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import yaml

p = argparse.ArgumentParser()
p.add_argument("--projects", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
revision = subprocess.check_output(
    ["git", "-C", str(a.projects), "rev-parse", "HEAD"], text=True
).strip()
assert revision == "23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a"
assert (a.checkpoint / "config.json").exists() and (
    a.checkpoint / "model_and_optim"
).is_dir()
runtime = Path(importlib.metadata.distribution("rslearn").locate_file("rslearn"))
report = {
    "project_revision": revision,
    "rslearn_version": importlib.metadata.version("rslearn"),
    "scope": "Pinned project configs, not a verified reproduction of paper experiments",
    "tasks": {},
    "runtime_sources": {},
}
for name in [
    "train/transforms/crop.py",
    "train/callbacks/freeze_unfreeze.py",
    "train/lightning_module.py",
    "train/tasks/per_pixel_regression.py",
    "models/olmoearth_pretrain/model.py",
]:
    source = runtime / name
    report["runtime_sources"][name] = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
for task in ["awf", "lfmc"]:
    source = a.projects / "olmoearth_run_data" / task / "model.yaml"
    config = yaml.safe_load(source.read_text())
    model = config["model"]["init_args"]
    data = config["data"]["init_args"]
    trainer = config["trainer"]
    callbacks = trainer["callbacks"]
    report["tasks"][task] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": model,
        "batch_size": data["batch_size"],
        "default_config": data["default_config"],
        "train_config": data["train_config"],
        "val_config": data["val_config"],
        "test_config": data["test_config"],
        "max_epochs": trainer["max_epochs"],
        "callbacks": callbacks,
        "test_points_to_validation": data["test_config"]["tags"]["split"] == "val",
    }
    candidate = copy.deepcopy(config)
    m = candidate["model"]["init_args"]["model"]["init_args"]
    encoder = m["encoder"][0]["init_args"]
    encoder.pop("model_id")
    encoder.update(
        checkpoint_path=str(a.checkpoint),
        embedding_size=1024,
        random_initialization=False,
    )
    if task == "awf":
        m["decoders"]["segment"][1]["init_args"]["in_channels"] = 1024
    else:
        m["decoder"][0]["init_args"]["in_channels"] = [[4, 1024]]
    candidate["trainer"]["logger"] = False
    candidate["trainer"]["callbacks"] = [
        c
        for c in candidate["trainer"]["callbacks"]
        if not c["class_path"].endswith(("LearningRateMonitor", "RslearnWriter"))
    ]
    # Offline local training only; prediction writer and external logging are removed.
    (a.output / f"{task}-large-candidate.yaml").write_text(
        yaml.safe_dump(candidate, sort_keys=False)
    )
report["candidate_status"] = (
    "Not launched. Requires dataset/runtime integration and evaluation randomness audit before scientific use."
)
report["candidate_changes"] = [
    "Base encoder replaced by own Large distributed checkpoint; native width 1024",
    "External logger and prediction writer removed",
    "Official transforms, heads, epochs, freeze/unfreeze, optimizer/scheduler settings retained",
    "test_config remains val and must never be reported as heldout test",
]
(a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "revision": revision,
            "rslearn": report["rslearn_version"],
            "tasks": {
                k: {
                    "batch_size": v["batch_size"],
                    "epochs": v["max_epochs"],
                    "test_is_val": v["test_points_to_validation"],
                }
                for k, v in report["tasks"].items()
            },
        },
        indent=2,
    )
)

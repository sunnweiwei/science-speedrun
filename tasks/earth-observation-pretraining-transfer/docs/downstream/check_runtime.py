"""CPU-only class resolution and evaluation crop behavior check."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import torch
import yaml
from rslearn.train.transforms.crop import Crop

p = argparse.ArgumentParser()
p.add_argument("--configs", type=Path, required=True)
a = p.parse_args()
classes = set()


def walk(x):
    if isinstance(x, dict):
        if "class_path" in x:
            classes.add(x["class_path"])
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)


for task in ["awf", "lfmc"]:
    walk(yaml.safe_load((a.configs / f"{task}-large-candidate.yaml").read_text()))
checks = {}
for name in sorted(classes):
    module, attr = name.rsplit(".", 1)
    getattr(importlib.import_module(module), attr)
    checks[name] = "resolved"
torch.manual_seed(42)
crop = Crop(crop_size=16)
samples = []
for _ in range(32):
    state = crop.sample_state((31, 31))
    samples.append([int(state["remove_from_left"]), int(state["remove_from_top"])])
assert len(set(map(tuple, samples))) > 1
fixed = Crop(crop_size=16, offset=(7, 7))
assert all(
    int(fixed.sample_state((31, 31))["remove_from_left"]) == 7 for _ in range(10)
)
report = {
    "class_resolution": checks,
    "minimal_version": importlib.metadata.version("olmoearth-pretrain-minimal"),
    "cuda_used": False,
    "random_crop_offsets": samples,
    "distinct_offsets": len(set(map(tuple, samples))),
    "fixed_offset_check": True,
    "limitation": "Import and transform checks only; not end-to-end dataset/model training validation.",
}
(a.configs / "runtime-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "classes_resolved": len(checks),
            "distinct_crop_offsets": report["distinct_offsets"],
            "fixed_offset_check": True,
        }
    )
)

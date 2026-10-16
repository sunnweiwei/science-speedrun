"""Turn a FAIRChem training checkpoint into a submission directory (model.pt + calculator.py).

    python export_submission.py --checkpoint RUN/checkpoints/final/inference_ckpt.pt --output /workspace/submission

As in uma-mp-finetune/export_model.py, FAIRChem 2.16's predictor loads the averaged-model slot,
so it is filled with one copy of the trained (non-EMA) weights. Training-only dataset aliases
are removed from the task specs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def export(checkpoint: Path, output: Path):
    import torch
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists():
        raise FileExistsError(output / "model.pt")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.ema_state_dict = {"module." + k: v for k, v in model.model_state_dict.items()}
    model.ema_state_dict["n_averaged"] = torch.tensor(1, dtype=torch.long)
    # Training-only config names (the "offeq" alias) are unknown to the saved backbone config; keep
    # only pretrained dataset names in the task specs so the stock predictor validates them.
    known = set(model.model_config["backbone"].get("dataset_list") or
                model.model_config["backbone"].get("dataset_mapping", {}))
    for task in model.tasks_config:
        task["datasets"] = [d for d in task["datasets"] if d in known]
    torch.save(model, output / "model.pt")
    shutil.copyfile(HERE / "calculator.py", output / "calculator.py")
    (output / "export.json").write_text(json.dumps({
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "weights": "model", "runtime": "fairchem-core==2.16.0", "task_name": "omat",
        "inference_settings": "default"}, indent=1))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(export(a.checkpoint, a.output))

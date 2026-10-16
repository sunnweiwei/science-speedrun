"""Export a self-contained checkpoint and its training-convention provenance."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from common import digest, write_json


def export(
    checkpoint: Path, data: Path, output: Path, weights="model", *, preparation: Path
):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        manifest.get("energy_convention") != "mp-pbe-raw-total"
        or manifest.get("corrections_applied") is not False
    ):
        raise ValueError("Unsupported energy convention")
    for filename, sha in manifest["files"].items():
        if digest(data / filename) != sha:
            raise ValueError("Training data checksum mismatch")
    from prepare_training import INITIALIZATION

    receipt = json.loads(preparation.read_text())
    if (
        receipt.get("initialization") != INITIALIZATION
        or receipt.get("initial_checkpoint_sha256") is not None
        or receipt.get("pretrained_weights_allowed") is not False
        or receipt.get("training_manifest_sha256") != digest(data / "manifest.json")
        or receipt.get("statistics_train_sha256") != manifest["files"]["train.jsonl"]
    ):
        raise ValueError("Missing or incompatible pretraining provenance")
    import torch

    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if weights == "model":
        # FAIRChem 2.16's public predictor unconditionally loads the averaged
        # model container. Populate it with one copy of the final trained
        # weights, so inference is exactly non-EMA without a custom loader.
        model.ema_state_dict = {
            "module." + key: value for key, value in model.model_state_dict.items()
        }
        model.ema_state_dict["n_averaged"] = torch.tensor(1, dtype=torch.long)
    if weights == "ema" and not model.ema_state_dict:
        raise ValueError("Checkpoint has no EMA weights")
    output.mkdir(parents=True)
    torch.save(model, output / "model.pt")
    shutil.copyfile(
        Path(__file__).resolve().parent.parent / "uma-mp-finetune/calculator.py",
        output / "calculator.py",
    )
    (output / "README.md").write_text(
        "# Random-initialized UMA-S pretraining\n\n"
        "Training provenance: preparation.json and training-manifest.json.\n"
        "Add data selection, compute, validation metrics and limitations before submission.\n"
    )
    shutil.copyfile(data / "manifest.json", output / "training-manifest.json")
    shutil.copyfile(preparation, output / "preparation.json")
    write_json(
        output / "energy-audit.json",
        dict(
            checkpoint_sha256=digest(output / "model.pt"),
            training_manifest_sha256=digest(data / "manifest.json"),
            energy_convention="mp-pbe-raw-total",
            corrections_applied=False,
            label_audit_passed=True,
            weights=weights,
            preparation_sha256=digest(preparation),
            note="Operator-reviewed label provenance; not an accuracy certificate or untrusted-submission attestation",
        ),
    )
    write_json(
        output / "export.json",
        dict(
            source_checkpoint_sha256=digest(checkpoint),
            weights=weights,
            initialization=receipt["initialization"],
            preparation_sha256=digest(preparation),
            runtime="fairchem-core==2.16.0",
            task_name="omat",
            inference_settings="default",
        ),
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "data", "output", "preparation"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--weights", choices=["model", "ema"], default="model")
    a = p.parse_args()
    print(export(a.checkpoint, a.data, a.output, a.weights, preparation=a.preparation))

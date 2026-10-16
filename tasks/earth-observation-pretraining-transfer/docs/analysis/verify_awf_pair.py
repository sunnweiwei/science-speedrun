"""Require a completed, exactly paired AWF integration pilot before full runs."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def verify(root, receipt):
    if receipt["stage"] != "integration pilot, not scientific evidence":
        raise ValueError("Wrong pilot receipt")
    outputs = [root / name for name in receipt["outputs"]]
    collected = []
    for output in outputs:
        artifacts = {
            name: read(output / f"{name}.json")
            for name in [
                "settings",
                "initialization",
                "data-coverage",
                "unfreeze-boundary",
                "epochs",
                "completion",
            ]
        }
        settings = artifacts["settings"]
        init = artifacts["initialization"]
        epochs = artifacts["epochs"]
        done = artifacts["completion"]
        if (
            not settings["pilot"]
            or settings["max_epochs"] != 22
            or settings["seed"] != receipt["head_and_loader_seed"]
        ):
            raise ValueError("Wrong pilot schedule or seed")
        if (
            settings["limit_train_batches"] != 2
            or settings["limit_val_batches"] != 2
            or settings["num_workers"] != 0
            or settings["validation_rng_seed"] != receipt["validation_rng_seed"]
        ):
            raise ValueError("Changed sampling or validation protocol")
        if settings["test_used"] or done["test_used"]:
            raise ValueError("Test outcomes must not enter this gate")
        for key, source in [
            ("train_source_sha256", "environment/downstream/train_awf.py"),
            ("adapter_source_sha256", "environment/downstream/portable_encoder.py"),
            ("config_sha256", "results/protocol-audit/awf-large-candidate.yaml"),
        ]:
            if settings[key] != receipt["source_sha256"][source]:
                raise ValueError("Pilot implementation differs from launch receipt")
        provenance = read(Path(settings["package"]) / "provenance.json")
        if init["encoder_file_sha256"] != provenance["encoder_sha256"]:
            raise ValueError("Wrong initial encoder")
        if (
            provenance.get("training_validity") == "invalid"
            or provenance.get("origin") == "official_public_pretrained_reference"
        ):
            raise ValueError("Ineligible package")
        if (
            artifacts["data-coverage"].get("train") != 1115
            or artifacts["data-coverage"].get("val") != 344
        ):
            raise ValueError("Incomplete released coverage")
        if artifacts["unfreeze-boundary"] != {
            "epoch": 20,
            "frozen_weights_unchanged": True,
        }:
            raise ValueError("Frozen encoder integrity failed")
        if (
            done["epochs"] != 22
            or not done["encoder_changed"]
            or done["final_encoder_state_digest"] == init["encoder_state_digest"]
        ):
            raise ValueError("Unfreezing did not update encoder")
        if [r["epoch"] for r in epochs] != list(range(22)):
            raise ValueError("Pilot did not cross the full unfreeze boundary")
        for row in epochs:
            if row["epoch"] < 20:
                if row["encoder_trainable_parameters"] != 0 or len(row["lr"]) != 1:
                    raise ValueError("Encoder was trainable before epoch20")
            elif row["encoder_trainable_parameters"] <= 0 or len(row["lr"]) != 2:
                raise ValueError("Encoder was not unfrozen at epoch20")
        collected.append(
            {"path": str(output), "artifacts": artifacts, "provenance": provenance}
        )
    trained, random = collected
    if (
        Path(trained["artifacts"]["settings"]["package"]).name
        != receipt["package_names"]["trained"]
        or Path(random["artifacts"]["settings"]["package"]).name
        != receipt["package_names"]["random"]
    ):
        raise ValueError("Pair differs from launch receipt")
    pt, pr = trained["provenance"], random["provenance"]
    if pt.get("origin") != "own_from_scratch_training" or not (
        pr.get("origin") == "random_initialization"
        or ("origin" not in pr and "checkpoint" in pr and pr["checkpoint"] is None)
    ):
        raise ValueError("Need own pretraining and a random control")
    for key in ["initialization_seed", "variant", "parameters"]:
        if pt[key] != pr[key]:
            raise ValueError("Unpaired encoder architecture or initialization seed")
    if (
        trained["artifacts"]["initialization"]["head_state_digest"]
        != random["artifacts"]["initialization"]["head_state_digest"]
    ):
        raise ValueError("Different initial downstream head")
    for key in [
        "dataset",
        "seed",
        "validation_rng_seed",
        "train_source_sha256",
        "adapter_source_sha256",
        "config_sha256",
    ]:
        if (
            trained["artifacts"]["settings"][key]
            != random["artifacts"]["settings"][key]
        ):
            raise ValueError("Different paired downstream protocol")
    return {
        "gate_passed": True,
        "scientific_gain_established": False,
        "test_used": False,
        "pair": {k: str(v) for k, v in zip(["trained", "random"], outputs)},
        "head_state_digest": trained["artifacts"]["initialization"][
            "head_state_digest"
        ],
        "source_sha256": receipt["source_sha256"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Preserve previous gate reports")
    result = verify(a.root, read(a.receipt))
    result["receipt_sha256"] = hashlib.sha256(a.receipt.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Validate LFMC integration evidence without selecting on pilot performance."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(receipt):
    if os.environ.get("EARTH_SLURM_PARTITION"):
        require(
            receipt["partition"] == os.environ["EARTH_SLURM_PARTITION"],
            "Wrong partition",
        )
    require(receipt["test_used"] is False, "Test exposure in receipt")
    require(len(receipt["outputs"]) == 2, "Expected exactly two outputs")
    paired = []
    for name, package in zip(receipt["outputs"], receipt["source_packages"]):
        output = Path(receipt["output_root"]) / name
        settings = read(output / "settings.json")
        done = read(output / "completion.json")
        epochs = read(output / "epochs.json")
        provenance = read(Path(settings["package"]) / "provenance.json")
        require(Path(settings["package"]).name == package, "Wrong package")
        require(settings["pilot"] is True, "Not a pilot")
        require(
            settings["test_used"] is False and done["test_used"] is False,
            "Test outcomes must not enter gate",
        )
        for key in ["seed", "epochs", "unfreeze_epoch", "global_batch", "microbatch"]:
            require(settings[key] == receipt[key], f"Changed {key}")
        require(settings["counts"] == receipt["expected_counts"], "Incomplete index")
        require(settings["batch_limit_per_epoch"] == 1, "Changed pilot coverage")
        for source, digest in settings["source_sha256"].items():
            require(
                receipt["source_sha256"]["environment/downstream/" + source] == digest,
                "Source differs from launch receipt",
            )
        require(
            set(settings["source_sha256"])
            == {
                "train_lfmc.py",
                "lfmc_model.py",
                "lfmc_batch.py",
                "lfmc_window.py",
                "masked_spatial.py",
                "portable_encoder.py",
            },
            "Missing source evidence",
        )
        require(
            settings["encoder_file_sha256"] == provenance["encoder_sha256"],
            "Wrong encoder hash",
        )
        require(provenance.get("training_validity") != "invalid", "Invalid pretraining")
        require(
            read(output / "unfreeze-boundary.json")
            == {"epoch": 20, "frozen_weights_unchanged": True},
            "Frozen integrity failed",
        )
        require(
            done["epochs"] == 22
            and done["encoder_changed"] is True
            and done["final_encoder_state_digest"] != settings["encoder_state_digest"],
            "Encoder did not update",
        )
        require([r["epoch"] for r in epochs] == list(range(22)), "Incomplete pilot")
        for row in epochs:
            require(
                row["encoder_trainable"] == (row["epoch"] >= 20),
                "Wrong freeze schedule",
            )
            require(
                len(row["lr"]) == (2 if row["epoch"] >= 20 else 1),
                "Wrong optimizer groups",
            )
            require(
                row["train_windows"] == 32 and row["validation"]["windows"] == 32,
                "Incomplete pilot batches",
            )
            require(row["validation"]["valid_pixels"] > 0, "Empty validation")
            require(
                all(
                    math.isfinite(v) and v >= 0
                    for v in [
                        row["train_loss"],
                        row["validation"]["mse"],
                        row["validation"]["mae"],
                        *row["lr"],
                    ]
                ),
                "Nonfinite or invalid metrics",
            )
        require(
            (output / "best.pt").is_file() and (output / "last.pt").is_file(),
            "Missing checkpoints",
        )
        paired.append((settings, provenance))
    (trained, pt), (random, pr) = paired
    require(pt.get("origin") == "own_from_scratch_training", "Not own pretraining")
    require(
        pr.get("origin") == "random_initialization"
        or ("origin" not in pr and "checkpoint" in pr and pr["checkpoint"] is None),
        "Not a random control",
    )
    for key in ["initialization_seed", "variant", "parameters"]:
        require(pt[key] == pr[key], "Unpaired encoder initialization/architecture")
    for key in [
        "head_state_digest",
        "index_sha256",
        "source_sha256",
        "seed",
        "training_crop",
        "validation_crop",
        "gpu",
    ]:
        require(trained[key] == random[key], f"Unpaired {key}")
    return {
        "gate_passed": True,
        "scientific_gain_established": False,
        "test_used": False,
        "outputs": receipt["outputs"],
        "head_state_digest": trained["head_state_digest"],
        "index_sha256": trained["index_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(read(args.receipt))
    result["receipt_sha256"] = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

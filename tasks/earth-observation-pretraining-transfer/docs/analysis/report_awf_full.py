"""Audit completed full AWF development pairs and plot all training epochs."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch



def moved(path):
    """Receipts name files by their pre-2026-10-16 repository paths."""
    for old, new in (("environment/downstream/", "docs/downstream/"), ("results/", "docs/analysis/"),
                     ("environment/starter/", "environment/baseline/starter/"), ("slurm/", "docs/slurm/")):
        if path.startswith(old):
            return new + path[len(old):]
    return path

def read(path):
    return json.loads(path.read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = read(args.receipt)
    require(
        receipt["planned_epochs"] == 100 and receipt["full_batches"], "Wrong schedule"
    )
    require(
        receipt["checkpoint_selection"] == "highest validation accuracy",
        "Changed checkpoint selection",
    )
    track = Path(__file__).resolve().parents[2]
    for path, sha in receipt["gate"]["source_sha256"].items():
        require(
            hashlib.sha256((track / moved(path)).read_bytes()).hexdigest() == sha,
            "Source changed",
        )
    collected = {}
    for plan in receipt["plans"]:
        out = Path(plan["output"])
        artifacts = {
            name: read(out / f"{name}.json")
            for name in (
                "settings",
                "initialization",
                "data-coverage",
                "unfreeze-boundary",
                "epochs",
                "completion",
            )
        }
        settings, init, epochs, done = (
            artifacts[k] for k in ("settings", "initialization", "epochs", "completion")
        )
        require(
            not settings["pilot"] and settings["max_epochs"] == 100,
            "Incomplete training recipe",
        )
        require(
            settings["limit_train_batches"] == settings["limit_val_batches"] == 1.0,
            "Limited batches",
        )
        require(
            settings["seed"] == receipt["head_and_loader_seed"]
            and settings["validation_rng_seed"] == receipt["validation_rng_seed"],
            "Changed RNG",
        )
        require(settings["package"] == plan["package"], "Changed package")
        require(
            not settings["test_used"]
            and not done["test_used"]
            and not receipt["test_used"],
            "Test used",
        )
        require(
            artifacts["data-coverage"] == {"train": 1115, "val": 344},
            "Incomplete split coverage",
        )
        require(
            artifacts["unfreeze-boundary"]
            == {"epoch": 20, "frozen_weights_unchanged": True},
            "Frozen encoder changed",
        )
        require(
            done["epochs"] == 100
            and done["encoder_changed"]
            and done["final_encoder_state_digest"] != init["encoder_state_digest"],
            "Unfreezing failed",
        )
        require([r["epoch"] for r in epochs] == list(range(100)), "Missing epochs")
        for row in epochs:
            require(
                row["encoder_trainable_parameters"]
                == (0 if row["epoch"] < 20 else 307770368),
                "Wrong freeze schedule",
            )
            require(
                len(row["lr"]) == (1 if row["epoch"] < 20 else 2),
                "Wrong optimizer groups",
            )
            require(
                all(math.isfinite(v) for v in row["metrics"].values()),
                "Nonfinite metric",
            )
        provenance = read(Path(plan["package"]) / "provenance.json")
        require(provenance.get("training_validity") != "invalid", "Invalid pretraining")
        require(
            init["encoder_file_sha256"] == provenance["encoder_sha256"],
            "Wrong encoder loaded",
        )
        for field, source in [
            ("train_source_sha256", "environment/downstream/train_awf.py"),
            ("adapter_source_sha256", "environment/downstream/portable_encoder.py"),
            ("config_sha256", "results/protocol-audit/awf-large-candidate.yaml"),
        ]:
            require(
                settings[field] == receipt["gate"]["source_sha256"][source],
                "Wrong runtime source",
            )
        best = max(epochs, key=lambda r: r["metrics"]["val_segment/accuracy"])
        checkpoints = [
            p for p in (out / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt"
        ]
        require(
            len(checkpoints) == 1 and (out / "checkpoints/last.ckpt").is_file(),
            "Missing checkpoints",
        )
        # mmap reads checkpoint metadata without allocating all optimizer/weight tensors.
        checkpoint = torch.load(
            checkpoints[0], map_location="cpu", weights_only=False, mmap=True
        )
        require(
            checkpoint["epoch"] == best["epoch"],
            "Best checkpoint epoch differs from metrics",
        )
        callbacks = [
            v
            for v in checkpoint["callbacks"].values()
            if isinstance(v, dict) and "best_model_score" in v
        ]
        require(len(callbacks) == 1, "Ambiguous checkpoint callback")
        score = float(callbacks[0]["best_model_score"])
        require(
            abs(score - best["metrics"]["val_segment/accuracy"]) < 1e-7,
            "Saved checkpoint score mismatch",
        )
        del checkpoint, callbacks
        gc.collect()
        collected[plan["kind"]] = {
            "artifacts": artifacts,
            "provenance": provenance,
            "best_validation_accuracy": score,
            "best_epoch_zero_based": best["epoch"],
            "final_validation_accuracy": epochs[-1]["metrics"]["val_segment/accuracy"],
            "checkpoint": str(checkpoints[0]),
            "checkpoint_bytes": checkpoints[0].stat().st_size,
            "artifact_sha256": {
                name: hashlib.sha256((out / f"{name}.json").read_bytes()).hexdigest()
                for name in artifacts
            },
        }
    trained, random = collected["trained"], collected["random"]
    require(
        trained["provenance"].get("origin") == "own_from_scratch_training",
        "Not own pretraining",
    )
    pr = random["provenance"]
    require(
        pr.get("origin") == "random_initialization"
        or ("origin" not in pr and "checkpoint" in pr and pr["checkpoint"] is None),
        "Not random control",
    )
    for key in ("initialization_seed", "variant", "parameters"):
        require(trained["provenance"][key] == pr[key], "Unpaired encoder")
    require(
        trained["artifacts"]["initialization"]["head_state_digest"]
        == random["artifacts"]["initialization"]["head_state_digest"],
        "Different heads",
    )
    for key in (
        "dataset",
        "seed",
        "validation_rng_seed",
        "num_workers",
        "train_source_sha256",
        "adapter_source_sha256",
        "config_sha256",
    ):
        require(
            trained["artifacts"]["settings"][key]
            == random["artifacts"]["settings"][key],
            "Unpaired downstream protocol",
        )
    report = {
        "paired_full_run_audit_passed": True,
        "scientific_gain_established": False,
        "test_used": False,
        "pretraining_seed": pr["initialization_seed"],
        "selection": receipt["checkpoint_selection"],
        "best_validation_gain_percentage_points": 100
        * (trained["best_validation_accuracy"] - random["best_validation_accuracy"]),
        "final_validation_gain_percentage_points": 100
        * (trained["final_validation_accuracy"] - random["final_validation_accuracy"]),
        "models": {
            k: {
                name: v
                for name, v in values.items()
                if name not in ("artifacts", "provenance")
            }
            for k, values in collected.items()
        },
        "limitations": "One development seed pair on344validation windows. Validation-selected checkpoints; no blind test or cross-seed stability proof. Checkpoint metadata audited; entire checkpoint files not rehashed here.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, label, color in [
        ("trained", "Own pretraining + fine-tuning", "#0072B2"),
        ("random", "Random initialization + fine-tuning", "#D55E00"),
    ]:
        rows = collected[name]["artifacts"]["epochs"]
        for ax, key, title in zip(
            axes,
            ["train_loss", "val_loss", "val_segment/accuracy"],
            ["Training loss", "Validation loss", "Validation accuracy (%)"],
        ):
            scale = 100 if "accuracy" in key else 1
            ax.plot(
                [r["epoch"] + 1 for r in rows],
                [scale * r["metrics"][key] for r in rows],
                label=label,
                color=color,
                linewidth=1.5,
            )
            ax.set(xlabel="Epoch", title=title)
            ax.axvline(20.5, color="gray", linestyle="--", linewidth=1)
            ax.grid(alpha=0.2)
        axes[2].scatter(
            collected[name]["best_epoch_zero_based"] + 1,
            collected[name]["best_validation_accuracy"] * 100,
            color=color,
            zorder=3,
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("OlmoEarth v1 Large: AWF development pair, 100 epochs")
    fig.text(
        0.5,
        0.01,
        "Dashed line: encoder unfreezing. One paired seed; validation only. Pretraining benefit not demonstrated.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(args.output / "learning-curve.png", dpi=180)
    fig.savefig(args.output / "learning-curve.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

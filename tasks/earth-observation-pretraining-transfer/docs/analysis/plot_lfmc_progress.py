"""Snapshot ongoing full LFMC curves; no checkpoint selection or target reads."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    histories, settings, hashes = {}, {}, {}
    for arm in ["trained", "random"]:
        root = args.root / f"full-{arm}-v1"
        payload = (root / "epochs.json").read_bytes()
        histories[arm] = json.loads(payload)
        hashes[arm] = hashlib.sha256(payload).hexdigest()
        settings[arm] = json.loads((root / "settings.json").read_text())
        assert settings[arm]["pilot"] is False and settings[arm]["test_used"] is False
        assert histories[arm] and [r["epoch"] for r in histories[arm]] == list(
            range(len(histories[arm]))
        )
        for row in histories[arm]:
            assert row["train_windows"] == 31175
            assert (
                row["validation"]["windows"]
                == row["validation"]["valid_pixels"]
                == 8262
            )
    for key in [
        "seed",
        "head_state_digest",
        "index_sha256",
        "source_sha256",
        "global_batch",
        "microbatch",
        "training_crop",
        "validation_crop",
    ]:
        assert settings["trained"][key] == settings["random"][key], key
    paired = [
        {
            "epoch": a["epoch"],
            "trained_mae": a["validation"]["mae"],
            "random_mae": b["validation"]["mae"],
            "mae_gain_random_minus_trained": b["validation"]["mae"]
            - a["validation"]["mae"],
        }
        for a, b in zip(histories["trained"], histories["random"])
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "histories": histories,
        "history_sha256": hashes,
        "paired_epochs": paired,
        "test_used": False,
        "stable_gain_established": False,
        "scope": "Ongoing single-seed development curves. Unequal completed epoch counts; compare paired epochs only. Checkpoint selection uses validation MSE, not minimum MAE. No external labels read.",
    }
    (args.output / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for arm, label in [
        ("trained", "Own pretraining"),
        ("random", "Paired random init"),
    ]:
        rows = histories[arm]
        epochs = [r["epoch"] + 1 for r in rows]
        for ax, values in zip(
            axes,
            [
                [r["train_loss"] for r in rows],
                [r["validation"]["mse"] for r in rows],
                [r["validation"]["mae"] for r in rows],
            ],
        ):
            ax.plot(epochs, values, marker="o", markersize=4, label=label)
    for ax, title in zip(axes, ["Training loss", "Validation MSE", "Validation MAE"]):
        ax.set(title=title, xlabel="Completed epoch (1-based)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "LFMC full development splits — early single-seed progress, not a final result"
    )
    fig.savefig(args.output / "learning-curve.png", dpi=160)
    fig.savefig(args.output / "learning-curve.pdf")
    print(
        json.dumps(
            {
                "completed_epochs": {k: len(v) for k, v in histories.items()},
                "paired_epochs": len(paired),
            }
        )
    )


if __name__ == "__main__":
    main()

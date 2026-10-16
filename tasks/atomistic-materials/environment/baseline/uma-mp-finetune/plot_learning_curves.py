"""Plot measured distributed train/development curves; never synthesize data."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path, partial=False):
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if not partial or i != len(lines) - 1:
                raise


def aggregate(records, world_size=8, partial=False):
    steps, evaluations = defaultdict(dict), defaultdict(dict)
    for path in sorted(Path(records).rglob("steps-rank-*.jsonl")):
        rank = int(path.stem.split("-")[-1])
        for row in read_jsonl(path, partial):
            steps[row["step"]][rank] = row
    for path in sorted(Path(records).rglob("validation-rank-*.jsonl")):
        rank = int(path.stem.split("-")[-1])
        for row in read_jsonl(path, partial):
            evaluations[row["step"]][rank] = row
    train = []
    for step, ranks in sorted(steps.items()):
        if set(ranks) != set(range(world_size)):
            if partial:
                continue
            raise ValueError(f"Missing train ranks at step {step}")
        loss = float(np.mean([r["loss"] for r in ranks.values()]))
        if not np.isfinite(loss):
            raise ValueError("Nonfinite train loss")
        train.append(
            {"step": step, "loss": loss, "learning_rate": ranks[0].get("learning_rate")}
        )
    val = []
    for step, ranks in sorted(evaluations.items()):
        if set(ranks) != set(range(world_size)):
            if partial:
                continue
            raise ValueError(f"Missing validation ranks at step {step}")
        metrics = dict(ranks[0]["metrics"])
        # Physical error metrics are already reduced by FAIRChem; the native
        # weighted validation loss is rank-local and needs this rank mean.
        metrics["val/loss"] = float(
            np.mean([r["metrics"]["val/loss"] for r in ranks.values()])
        )
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite validation metric")
        val.append({"step": step, "weights": ranks[0]["weights"], "metrics": metrics})
    return train, val


def plot(
    records,
    output,
    *,
    expected_steps=46875,
    steps_per_epoch=15625,
    partial=False,
    title="UMA-S-1.1 adaptation · 500k frames · 8×H100",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = aggregate(records, partial=partial)
    if not train or not val:
        raise ValueError("Need real training and validation observations")
    if not partial and (
        [r["step"] for r in train] != list(range(1, expected_steps + 1))
        or val[-1]["step"] != expected_steps
    ):
        raise ValueError("Incomplete final learning-curve coverage")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = min(val, key=lambda r: r["metrics"]["val/loss"])
    result = {
        "complete": not partial,
        "observed_steps": train[-1]["step"],
        "expected_steps": expected_steps,
        "steps_per_epoch": steps_per_epoch,
        "best_development_step": best["step"],
        "best_development_loss": best["metrics"]["val/loss"],
        "last_development": val[-1],
        "train": train,
        "development": val,
        "loss_definition": "FAIRChem normalized weighted E/F/S objective; arithmetic mean across all eight ranks",
        "note": "Development metrics use supplied geometries, not physical stability/phonon/elasticity evaluations.",
    }
    (output / "learning-curves.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    x = np.asarray([r["step"] / steps_per_epoch for r in train])
    y = np.asarray([r["loss"] for r in train])
    vx = np.asarray([r["step"] / steps_per_epoch for r in val])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3), layout="constrained")
    ax = axes[0, 0]
    stride = max(1, len(x) // 2400)
    ax.plot(
        x[::stride],
        y[::stride],
        color="#1966b3",
        alpha=0.15,
        lw=0.7,
        label="Training, raw",
    )
    window = min(200, len(y))
    smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    ax.plot(
        x[window - 1 :],
        smooth,
        color="#1966b3",
        lw=1.8,
        label=f"Training, {window}-step mean",
    )
    ax.plot(
        vx,
        [r["metrics"]["val/loss"] for r in val],
        color="#c45c16",
        marker="o",
        ms=3.4,
        lw=1.6,
        label="Development",
    )
    ax.set(
        title="Weighted loss (log scale)",
        ylabel="Normalized E/F/S objective",
        yscale="log",
    )
    ax.legend(frameon=False, fontsize=9)
    specs = [
        ("energy,per_atom_mae", 1000, "Energy MAE", "meV / atom"),
        ("forces,mae", 1000, "Force MAE", "meV / Å"),
        ("stress,mae", 160.21766208, "Stress MAE", "GPa"),
    ]
    for ax, (suffix, scale, name, unit) in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]], specs
    ):
        values = []
        for row in val:
            keys = [k for k in row["metrics"] if k.endswith("," + suffix)]
            if len(keys) != 1:
                raise ValueError(f"Missing/ambiguous metric {suffix}: {keys}")
            values.append(row["metrics"][keys[0]] * scale)
        ax.plot(vx, values, color="#c45c16", marker="o", ms=3.4, lw=1.6)
        ax.set(title=name, ylabel=unit, ylim=(0, max(values) * 1.1 or 1))
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18, which="major")
        ax.set_xlim(0, max(0.1, x[-1]))
        for epoch in range(1, int(x[-1]) + 1):
            ax.axvline(epoch, color="#777777", alpha=0.2, lw=0.8)
    status = (
        f"{train[-1]['step']:,} / {expected_steps:,} steps · 5,000 development frames"
    )
    if partial:
        status = "IN PROGRESS · " + status
    fig.suptitle(title + "\n" + status, fontsize=15, fontweight="bold")
    fig.savefig(output / "loss-curves.png", dpi=180)
    fig.savefig(output / "loss-curves.svg")
    plt.close(fig)
    summary = {k: v for k, v in result.items() if k not in ["train", "development"]}
    (output / "curve-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    print(json.dumps(plot(args.records, args.output, partial=args.partial), indent=2))

"""Plot paired seed gains and joint seed/spatial uncertainty from saved analysis."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--comparison", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
report = json.loads(a.comparison.read_text())
seeds = report["initialization_seeds"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for task, ax in zip(["awf", "lfmc"], axes):
    row = report["tasks"][task]
    scale = 100 if task == "awf" else 1
    gains = np.array(row["per_seed_gain"]) * scale
    ax.scatter(range(len(seeds)), gains, color="#2274a5", s=45, zorder=3)
    labels = [str(seed) for seed in seeds]
    summaries = [(row, "Mean vs\nrandom", "#2274a5")]
    if "raw_baseline_comparison" in row:
        summaries.append((row["raw_baseline_comparison"], "Mean vs\nraw", "#666666"))
    for index, (summary, label, color) in enumerate(summaries, start=len(seeds)):
        mean = summary["mean_seed_gain"] * scale
        low, high = (
            np.array(summary["seed_and_spatial_bootstrap_95_percent_interval"]) * scale
        )
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="s",
            capsize=5,
            color=color,
        )
        labels.append(label)
    ax.axhline(0, color="#a84432", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(
        "Accuracy gain (percentage points)" if task == "awf" else "MAE reduction"
    )
    state = "met" if row["three_seed_criterion_met"] else "not met"
    ax.set_title(f"{task.upper()}: declared criterion {state}")
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("OlmoEarth v1 Large: locked-recipe validation confirmation")
fig.text(
    0.5,
    0.02,
    "Positive favors training. Error bars: joint seed/spatial 95% intervals. Recipe chosen on development seed12536.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=160)

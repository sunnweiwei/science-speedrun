"""Collect actual logged training metrics, without interpolating missing steps."""

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=Path, required=True)
p.add_argument("--jobs", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
jobs = json.loads(a.jobs.read_text())
runs = [
    {"job": plan["training_job"], "run": Path(plan["settings"]["OLMO_RUN"]).name}
    for plan in jobs["corrected_v2_development_runs"]
]
metrics = ["train/ModalityPatchDisc", "train/InfoNCE", "optim/total grad norm"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
report = {}
for run in runs:
    text = (a.data_root / f"train-{run['job']}.log").read_text()
    rows, step = {}, None
    for line in text.splitlines():
        header = re.search(r"console_logger.*\[step=(\d+)/", line)
        if header:
            step = int(header.group(1))
        elif step is not None and line.startswith("    ") and "=" in line:
            key, value = line.strip().rsplit("=", 1)
            if key in metrics or key == "gpu_memory/GPU reserved mem (GiB)":
                rows.setdefault(step, {})[key] = float(value.replace(",", ""))
    for metric, axis in zip(metrics, axes):
        points = sorted(
            (step, row[metric]) for step, row in rows.items() if metric in row
        )
        if points:
            x, y = map(np.array, zip(*points))
            # Display 100-update bins; preserve all raw observations in JSON.
            bins = x // 100
            axis.plot(
                [x[bins == b].mean() for b in np.unique(bins)],
                [y[bins == b].mean() for b in np.unique(bins)],
                label=run["run"],
            )
        axis.set_title(metric.split("/")[-1])
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.2)
    report[run["run"]] = {
        "job": run["job"],
        "nonfinite_batch_warnings": text.count("NaN or Inf detected in loss"),
        "logged_metrics": rows,
    }
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
fig.suptitle("OlmoEarth v1 Large (recomputation v2): metrics within100-update bins")
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
a.output.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output / "training-development.png", dpi=160)
(a.output / "training-development.json").write_text(json.dumps(report, indent=2))
print(
    {
        name: {
            "last_logged_step": max(row["logged_metrics"]),
            "nonfinite_batch_warnings": row["nonfinite_batch_warnings"],
        }
        for name, row in report.items()
    }
)

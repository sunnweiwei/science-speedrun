"""Descriptive representation diagnostics on AWF training rows only."""

import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=Path, required=True)
p.add_argument("--names", nargs="+", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = json.loads((a.data_root / "eval/awf/labels.json").read_text())
ids = [r["id"] for r in rows if r["split"] == "train"]
report = {}
for name in a.names:
    data = np.load(
        a.data_root / "features" / f"{name}-awf/features.npz", allow_pickle=False
    )
    lookup = {key: i for i, key in enumerate(data["ids"].tolist())}
    x = data["features"][[lookup[key] for key in ids]].astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    z = x / np.maximum(norms[:, None], 1e-12)
    n = len(x)
    cosine = (np.linalg.norm(z.sum(axis=0)) ** 2 - n) / (n * (n - 1))
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    variance = singular**2
    variance /= variance.sum()
    effective_rank = np.exp(-(variance * np.log(np.maximum(variance, 1e-300))).sum())
    report[name] = {
        "training_rows": n,
        "mean_feature_norm": float(norms.mean()),
        "median_channel_std": float(np.median(x.std(axis=0))),
        "mean_pair_cosine": float(cosine),
        "centered_covariance_effective_rank": float(effective_rank),
    }
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

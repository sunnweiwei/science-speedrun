"""Descriptive train/validation AWF representation diagnostics; no test access."""

import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = json.loads((a.root / "eval/awf/labels.json").read_text())
assert all(r["split"] in ["train", "val"] for r in rows)
names = [
    "large-random-eval-12536",
    "large-v2-run005-step6000",
    "large-random-eval-12539",
    "large-v2-confirm2-12539-step6000",
    "large-random-eval-12540",
    "large-v2-confirm2-12540-step6000",
]
report = {
    "scope": "Descriptive AWF train/val only; no hyperparameter selection or causal claim",
    "models": {},
}
for name in names:
    f = np.load(a.root / "features" / f"{name}-awf/features.npz", allow_pickle=False)
    lookup = {v: i for i, v in enumerate(f["ids"].tolist())}
    assert len(lookup) == len(rows)
    result = {}
    for split in ["train", "val"]:
        chosen = [r for r in rows if r["split"] == split]
        x = f["features"][[lookup[r["id"]] for r in chosen]].astype(np.float64)
        y = np.array([r["label"] for r in chosen])
        center = x.mean(0)
        xc = x - center
        eigen = np.linalg.eigvalsh(xc @ xc.T)
        eigen = np.maximum(eigen, 0)
        probs = eigen[eigen > 0] / eigen.sum()
        within = sum(
            float(((x[y == c] - x[y == c].mean(0)) ** 2).sum()) for c in np.unique(y)
        )
        between = float((xc**2).sum()) - within
        result[split] = {
            "rows": len(x),
            "finite": bool(np.isfinite(x).all()),
            "mean_feature_norm": float(np.linalg.norm(x, axis=1).mean()),
            "total_centered_variance": float(np.var(x, axis=0).sum()),
            "effective_covariance_rank": float(np.exp(-(probs * np.log(probs)).sum())),
            "between_within_class_scatter_ratio": between / within,
        }
    report["models"][name] = result
    print(name, result, flush=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")

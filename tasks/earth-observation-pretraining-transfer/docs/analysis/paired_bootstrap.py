"""LFMC uncertainty, resampling sites rather than correlated dates."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=Path, required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--random-name", default="random")
p.add_argument("--selection-file", type=Path)
p.add_argument("--split", choices=["val", "test"], default="val")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.split == "test":
    selection = json.loads(
        (a.selection_file or a.data_root / "selection.json").read_text()
    )
    if a.candidate != selection["selected"]:
        raise ValueError("Test analysis is restricted to the locked checkpoint")
provenance = [
    json.loads((a.data_root / "packages" / name / "provenance.json").read_text())
    for name in (a.random_name, a.candidate)
]
if any(
    p.get("training_validity") == "invalid"
    or p.get("origin") == "official_public_pretrained_reference"
    for p in provenance
):
    raise ValueError("Ineligible training artifact")
if provenance[0]["checkpoint"] is not None or provenance[1]["checkpoint"] is None:
    raise ValueError("Expected paired random and own trained checkpoint")
for key in ("variant", "parameters", "initialization_seed", "infer_sha256"):
    if provenance[0][key] != provenance[1][key]:
        raise ValueError(f"Unpaired {key}")
if a.split == "test" and provenance[1]["encoder_sha256"] != selection["encoder_sha256"]:
    raise ValueError("Candidate provenance does not match locked weight hash")
rows = json.loads((a.data_root / "eval/lfmc/labels.json").read_text())
y = np.array([r["label"] for r in rows])
tr = np.array([r["split"] == "train" for r in rows])
va = np.array([r["split"] == a.split for r in rows])
errors = []
for name in [a.random_name, a.candidate]:
    result = json.loads((a.data_root / "results" / f"{name}-lfmc.json").read_text())
    data = np.load(result["features"], allow_pickle=False)
    lookup = {str(value): i for i, value in enumerate(data["ids"])}
    x = data["features"][[lookup[r["id"]] for r in rows]].astype(np.float64)
    model = make_pipeline(
        StandardScaler(), Ridge(alpha=result["probe"]["selected_parameter"])
    )
    model.fit(x[tr], y[tr])
    errors.append(np.abs(model.predict(x[va]) - y[va]))
    if a.split == "val":
        assert abs(errors[-1].mean() - result["probe"]["validation"]["mae"]) < 1e-7

sites = np.array(
    [tuple(round(v, 6) for v in r["latlon"]) for r in rows if r["split"] == a.split]
)
unique, inverse = np.unique(sites, axis=0, return_inverse=True)
gain = errors[0] - errors[1]
totals = np.bincount(inverse, weights=gain)
counts = np.bincount(inverse)
rng = np.random.default_rng(42)
samples = rng.integers(0, len(unique), size=(5000, len(unique)))
boot = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
report = {
    "candidate": a.candidate,
    "random_baseline": a.random_name,
    "split": a.split,
    "rows": int(va.sum()),
    "sites": len(unique),
    "bootstrap_replicates": 5000,
    "seed": 42,
    "mae_reduction_vs_random": float(gain.mean()),
    "site_bootstrap_95_percent_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
    "interpretation": "Positive reduction favors training; conditional on one model initialization, not training-seed uncertainty",
}
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

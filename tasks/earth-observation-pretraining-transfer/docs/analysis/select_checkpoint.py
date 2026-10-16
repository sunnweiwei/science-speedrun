"""Apply the recorded validation-only checkpoint policy before test scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=Path, required=True)
p.add_argument(
    "--policy",
    type=Path,
    help="Explicit policy file; defaults to DATA/selection-policy.json",
)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error("selection output already exists; preserve the recorded selection")
policy_path = a.policy or a.data_root / "selection-policy.json"
policy = json.loads(policy_path.read_text())
primary = policy.get("primary_task", "joint")
if primary not in {"joint", "lfmc"}:
    raise ValueError("Unknown primary_task in selection policy")
context_path = a.data_root / "execution-context.json"
context = json.loads(context_path.read_text()) if context_path.exists() else None
if (
    policy.get("execution_context") is not None
    and policy["execution_context"] != context
):
    raise ValueError("Policy execution context does not match evaluation root")
candidates = []
for name in policy["candidates"]:
    provenance = json.loads(
        (a.data_root / "packages" / name / "provenance.json").read_text()
    )
    if provenance.get("training_validity") == "invalid":
        raise ValueError(f"Invalid training candidate: {name}")
    if provenance.get("origin") == "official_public_pretrained_reference":
        raise ValueError("Public pretrained weights cannot be selected as our training")
    if provenance.get("checkpoint") is None:
        raise ValueError(f"Candidate has no own training checkpoint: {name}")
    metrics = {}
    hashes = {}
    for task in ["awf", "lfmc"]:
        path = a.data_root / "results" / f"{name}-{task}.json"
        result = json.loads(path.read_text())
        if context is not None and (
            Path(result["features"]).resolve()
            != (a.data_root / "features" / f"{name}-{task}" / "features.npz").resolve()
            or Path(result["validation_predictions"]).resolve()
            != (a.data_root / "results" / f"{name}-{task}-predictions.npz").resolve()
        ):
            raise ValueError("Cannot select across different execution contexts")
        if "test" in result["probe"]:
            raise ValueError(
                "Checkpoint selection must use the validation-only reports"
            )
        metrics[task] = result
        hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    accuracy = metrics["awf"]["probe"]["validation"]["accuracy"]
    mae = metrics["lfmc"]["probe"]["validation"]["mae"]
    constant = metrics["lfmc"]["constant_val"]["mae"]
    candidates.append(
        {
            "name": name,
            "step": int(name.split("-step")[1]),
            "awf_validation_accuracy": accuracy,
            "lfmc_validation_mae": mae,
            "selection_score": 0.5 * (accuracy + max(0.0, 1.0 - mae / constant)),
            "metric_file_sha256": hashes,
        }
    )
winner = (
    min(candidates, key=lambda c: (c["lfmc_validation_mae"], c["name"]))
    if primary == "lfmc"
    else max(candidates, key=lambda c: (c["selection_score"], c["step"], c["name"]))
)
package = a.data_root / "packages" / winner["name"]
provenance = json.loads((package / "provenance.json").read_text())
weight_hash = hashlib.sha256((package / "encoder.pt").read_bytes()).hexdigest()
assert weight_hash == provenance["encoder_sha256"]
report = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "selected": winner["name"],
    "package": str(package),
    "encoder_sha256": weight_hash,
    "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    "primary_task": primary,
    "ranking_only": True,
    "execution_context": context,
    "candidates": candidates,
    "test_metrics_used": False,
}
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

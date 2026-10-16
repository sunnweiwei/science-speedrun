"""Freeze a completed zero-control measurement as a reconstructed Eq.36 target."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cavity_reward import CavityReward


def calibrate(run, output, max_mean_drift=.25):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError("Do not replace an existing calibration")
    protocol = json.loads((run / "protocol.json").read_text())
    result = json.loads((run / "diagnostic.json").read_text())
    if (protocol["action"] != "zero" or protocol["minimum_calibration_cycles"] < 1000
            or not result.get("estimated_cycle_coverage_met")
            or result.get("max_block_mean_drift_in_std", float("inf")) > max_mean_drift):
        raise ValueError("Need >=1000 measured tail cycles and acceptable blockwise mean drift")
    with np.load(run / "raw-measurements.npz") as data:
        samples = data["observations"]
    if samples.shape[0] != protocol["steps"] or samples.shape[0] != result["samples"]:
        raise ValueError("Incomplete measurement budget")
    tail = samples[len(samples) // 2:]
    mean, std = tail.mean(axis=0), tail.std(axis=0)
    np.testing.assert_allclose(mean, result["mean"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(std, result["std_population"], rtol=1e-12, atol=1e-15)
    reward = CavityReward(mean, std)
    mean_score = float(np.mean([reward(x) for x in tail]))
    np.testing.assert_allclose(mean_score, -samples.shape[1], rtol=1e-12)
    document = {"status": "reconstructed-cavity-calibration-not-paper-certified",
        "binding": {"reynolds": protocol["reynolds"], "probe_locations": protocol["probe_locations"],
                    "observation_types": protocol["observations"], "substeps": protocol["substeps_per_action"]},
        "mean": mean.tolist(), "std_population": std.tolist(),
        "measurement": {"samples": len(samples), "tail_samples": len(tail),
                        "estimated_tail_cycles": result["estimated_tail_cycles"],
                        "tail_peak_count": result["tail_peak_count"],
                        "max_block_mean_drift_in_std": result["max_block_mean_drift_in_std"],
                        "organizer_mean_drift_limit": max_mean_drift,
                        "tail_mean_eq36_reward": mean_score},
        "source_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                          for name in ("protocol.json", "diagnostic.json", "raw-measurements.npz", "source.py")},
        "probe_provenance": protocol["probe_provenance"],
        "unresolved": ["exact figure-specific target vectors, initial condition and aggregation",
                       "paper prose geometry differs from available resources"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mean-drift", type=float, default=.25)
    args = parser.parse_args()
    if not np.isfinite(args.max_mean_drift) or args.max_mean_drift <= 0:
        parser.error("Mean-drift limit must be finite and positive")
    print(json.dumps(calibrate(args.run, args.output, args.max_mean_drift), indent=2))

"""Export measured program and reference points from paired development or private details JSON."""
import argparse
import csv
import json
from pathlib import Path


def points(paths):
    rows = []
    workload = None
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("protocol") != "rfd4-program-v1" or "comparison" not in report:
            raise ValueError(f"{path}: expected a completed paired report")
        device = report["device"]
        identity = (report["workload_sha256"], report.get("suite_sha256"), device["uuid"], device.get("driver"))
        if workload is not None and identity != workload:
            raise ValueError("Request sets, seeds or measurement devices differ")
        workload = identity
        for label, point, complete in (("submission", report, report["comparison"]["program_complete"]),
                                       (report["reference_name"], report["reference"], report["comparison"]["reference_complete"])):
            rows.append({"run": str(path), "method": label, "budget_seconds_per_slot": report["seconds_per_slot"],
                         "package_sha256": report["submission_sha256"] if label == "submission" else report["reference_sha256"],
                         "complete": complete,
                         "clustered_pass_pct": 100 * point["reward"], "gpu_seconds": point["total_gpu_seconds"],
                         "generation_attempts_per_gpu_hour": point["generation_attempts_per_gpu_hour"],
                         "end_to_end_attempts_per_gpu_hour": point["attempts_per_gpu_hour"],
                         "distinct_clusters_per_gpu_hour": point["clusters_per_allocated_gpu_hour"]})
    for row in rows:
        cost, quality = row["gpu_seconds"], row["clustered_pass_pct"]
        row["on_observed_frontier"] = bool(row["complete"]) and not any(
            p["complete"] and p["gpu_seconds"] <= cost and p["clustered_pass_pct"] >= quality and
            (p["gpu_seconds"] < cost or p["clustered_pass_pct"] > quality) for p in rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = points(args.reports)
    with args.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

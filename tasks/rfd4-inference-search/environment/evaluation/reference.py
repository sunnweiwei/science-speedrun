"""Package a pinned human baseline for the executable submission interface."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from policy import load_policy

BASELINES = Path(__file__).resolve().parents[1] / "baseline"
POLICIES = ("stochastic-tsr", "geometry-beam", "two-wide-beam")
NAMES = POLICIES + ("native-low", "native-high")


def historical_points(lengths, slots):
    """Return the original public measurements for this roster and sample count."""
    if sorted(lengths) != [100, 150, 250] or slots not in (10, 50):
        return {}
    size = "development" if slots == 10 else "paper-size"
    points = {}
    for name in POLICIES:
        record = json.loads((BASELINES / name / "reproduction" / f"{size}.json").read_text())
        rows = record["by_length"].values()
        points[name] = {"source": f"baseline/{name}/reproduction/{size}.json", "device": record["device"],
                        "slots": sum(r["slots"] for r in rows), "reward": record["reward"],
                        "passing_clusters": sum(r["passing_clusters"] for r in rows),
                        "total_gpu_seconds": sum(r["generation_seconds"] for r in rows) + record["af3_seconds"],
                        "clusters_per_allocated_gpu_hour": record["clusters_per_allocated_gpu_hour"]}
    sources = [
        ("monomer_beam/matched_30slot_summary.csv", "method", "clusters_seed_314159",
         "end_to_end_attempts_per_gpu_h", "end_to_end_clusters_per_gpu_h_seed_314159"),
    ] if slots == 10 else [
        ("monomer_beam/scaling_150slot_summary.csv", "method", "passing_clusters",
         "end_to_end_attempts_per_gpu_h", "end_to_end_clusters_per_gpu_h"),
        ("s15/summary.csv", "sampler", "passing_clusters", "generation_and_af3_attempts_per_gpu_h",
         "generation_and_af3_passing_clusters_per_gpu_h"),
        ("s15/base_schedule_summary.csv", "sampler", "passing_clusters", "generation_and_af3_attempts_per_gpu_h",
         "generation_and_af3_passing_clusters_per_gpu_h"),
    ]
    for source, method, count, throughput, yield_field in sources:
        path = BASELINES.parent / "references" / source
        if path.is_file():
            with path.open() as f:
                for row in csv.DictReader(f):
                    attempts = int(row["slots"])
                    clusters = int(row[count])
                    points[f"{source}/{row[method]}"] = {
                        "source": f"references/{source}", "device": "NVIDIA RTX A6000", "slots": attempts,
                        "reward": clusters / attempts, "passing_clusters": clusters,
                        "total_gpu_seconds": 3600 * attempts / float(row[throughput]),
                        "clusters_per_allocated_gpu_hour": float(row[yield_field])}
    return points


def package_reference(name, destination):
    if name not in NAMES:
        raise ValueError(f"Unknown reference: {name}")
    if name in POLICIES:
        return package_policy(BASELINES / name, destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Choose an empty reference package directory")
    (destination / "reference-preset.json").write_text(json.dumps(name.removeprefix("native-")) + "\n")
    shutil.copyfile(BASELINES / "program/infer.py", destination / "infer.py")
    return destination


def package_policy(source, destination):
    load_policy(source)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Choose an empty policy adapter directory")
    shutil.copyfile(source / "policy.json", destination / "policy.json")
    shutil.copyfile(Path(__file__).with_name("policy.py"), destination / "reference_policy.py")
    shutil.copyfile(Path(__file__).with_name("native_worker.py"), destination / "native_worker.py")
    shutil.copyfile(BASELINES / "program/infer.py", destination / "infer.py")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=NAMES)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package_reference(args.name, args.destination)

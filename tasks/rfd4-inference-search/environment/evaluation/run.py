import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from generate import generate
from policy import LENGTHS, InvalidPolicy, load_policy
from score import score


def run(package, assets, work, slots, seed, lengths):
    plans = load_policy(package)
    required = [assets / "rfd4/config.json", assets / "rfd4/model.safetensors", assets / "af3/af3.bin.zst",
                *(assets / "monomers" / f"monomer-{size}.cif.zst" for size in LENGTHS.values()),
                Path(os.environ.get("AF3_SCRIPT", "/opt/af3/run_alphafold.py"))]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing runtime asset: {path}")
    for executable in (assets / "foldseek", Path(os.environ.get("AF3_PYTHON", "/opt/af3env/bin/python"))):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Missing evaluator executable: {executable}")
    outputs, costs = {}, {}
    for length in lengths:
        folder = work / str(length)
        outputs[length], costs[length] = generate(plans[length], length, slots, seed + length, assets, folder)
        (folder / "generation.json").write_text(json.dumps(costs[length], indent=2) + "\n")
    report = score(outputs, slots, assets, work / "evaluation", seed)
    report.update(protocol="rfd4-monomer-v1", evaluated_lengths=list(lengths))
    for length, timing in costs.items():
        report["by_length"][length].update(timing)
    seconds = report["af3_seconds"] + sum(c["generation_seconds"] for c in costs.values())
    report["clusters_per_allocated_gpu_hour"] = 3600 * sum(
        v["passing_clusters"] for v in report["by_length"].values()) / seconds
    return report


def harbor_metrics(report):
    """Harbor 0.23 requires every value in reward.json to be numeric."""
    values = {"reward": report["reward"]}
    if "invalid_submission" in report:
        return dict(values, invalid_submission=1)
    for key in ("af3_seconds", "clusters_per_allocated_gpu_hour"):
        values[key] = report[key]
    for length, row in report["by_length"].items():
        for name, value in row.items():
            values[f"length_{length}_{name}"] = int(value) if isinstance(value, bool) else value
    return values


def write_result(target, report):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    published = Path(f.name)
    published.chmod(0o644)
    published.replace(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    p.add_argument("--assets", type=Path, default=Path("/workspace/data/assets"))
    p.add_argument("--work", type=Path, required=True, help="New or empty scratch directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-seed", type=Path, help="Controller seed file for a 150-slot final replay")
    p.add_argument("--paper-size", action="store_true", help="Public 150-slot development run")
    p.add_argument("--length", type=int, choices=tuple(LENGTHS), help="Evaluate one development length")
    args = p.parse_args()
    if args.length and args.private_seed:
        p.error("Private evaluation requires all three lengths")
    work, target = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()) or target.exists():
        p.error("Use a fresh work directory and output path")
    seed = 727000
    if args.private_seed:
        data = args.private_seed.read_bytes()
        if len(data) != 32:
            raise RuntimeError("Private seed must contain 32 bytes")
        seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "big") % (2**31 - 1024)
    slots = 50 if args.private_seed or args.paper_size else 10
    try:
        lengths = [args.length] if args.length else tuple(LENGTHS)
        report = run(args.submission, args.assets.resolve(), work, slots, seed, lengths)
        if args.private_seed:
            report["suite_sha256"] = hashlib.sha256(data).hexdigest()
        else:
            report["development_seed"] = seed
    except InvalidPolicy as exc:
        report = {"reward": 0.0, "invalid_submission": str(exc)}
    if args.private_seed:
        write_result(target.with_name("details.json"), report)
        report = harbor_metrics(report)
    write_result(target, report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

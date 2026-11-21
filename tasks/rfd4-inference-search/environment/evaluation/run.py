import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from generate import generate
from policy import LENGTHS, SECONDS_PER_SLOT, InvalidPolicy, load_policy
from score import score
from program import gpu_device, package_digest, run_program, verifier_runtime
from requests_io import suite_requests
from reference import NAMES, historical_points, package_policy, package_reference


def run(package, assets, work, slots, seed, lengths, *, private=False, final_seed=None,
        seconds_per_slot=SECONDS_PER_SLOT, requests=None):
    package = Path(package).resolve()
    program = (package / "infer.py").is_file() or (package / "infer.py").is_symlink()
    plans = None if program else load_policy(package)
    if not program and any(n not in LENGTHS for n in lengths):
        raise InvalidPolicy("Use an infer.py package for requests outside the legacy suite")
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
        if program:
            outputs[length], costs[length] = run_program(package, assets, folder, length, slots,
                                                         (seed + length) % 2**31, private=private,
                                                         seconds_per_slot=seconds_per_slot,
                                                         descriptor=requests[length] if requests else None)
        else:
            outputs[length], costs[length] = generate(plans[length], length, slots, seed + length, assets, folder,
                                                      seconds_per_slot=seconds_per_slot)
        (folder / "generation.json").write_text(json.dumps(costs[length], indent=2) + "\n")
    report = score(outputs, slots, assets, work / "evaluation", final_seed if program and final_seed is not None else seed)
    report.update(protocol="rfd4-program-v1" if program else "rfd4-monomer-v1", evaluated_lengths=list(lengths),
                  seconds_per_slot=seconds_per_slot)
    for length, timing in costs.items():
        report["by_length"][length].update(timing)
    report["generation_seconds"] = sum(c["generation_seconds"] for c in costs.values())
    report["generation_attempts_per_gpu_hour"] = 3600 * slots * len(lengths) / report["generation_seconds"]
    seconds = report["generation_seconds"] + report["af3_seconds"]
    report["total_gpu_seconds"] = seconds
    report["attempts_per_gpu_hour"] = 3600 * slots * len(lengths) / seconds
    report["clusters_per_allocated_gpu_hour"] = 3600 * sum(
        v["passing_clusters"] for v in report["by_length"].values()) / seconds
    return report


def compare(report, reference):
    quality_delta = 100 * (report["reward"] - reference["reward"])
    ratio = report["total_gpu_seconds"] / reference["total_gpu_seconds"]
    reference_complete = all(not row["timed_out"] and row["refolded"] == row["slots"]
                             for row in reference["by_length"].values())
    program_complete = all(not row["timed_out"] and row["refolded"] == row["slots"]
                           for row in report["by_length"].values())
    comparable = reference_complete and program_complete
    return {"quality_delta_pp": quality_delta, "total_cost_ratio": ratio,
            "yield_delta_per_gpu_hour": report["clusters_per_allocated_gpu_hour"] -
                                        reference["clusters_per_allocated_gpu_hour"],
            "reference_complete": int(reference_complete), "program_complete": int(program_complete),
            "dominates_reference": int(comparable and quality_delta >= 0 and ratio <= 1
                                       and (quality_delta > 0 or ratio < 1)),
            "reference_dominates": int(comparable and quality_delta <= 0 and ratio >= 1
                                       and (quality_delta < 0 or ratio > 1))}


def harbor_metrics(report):
    """Harbor 0.23 requires every value in reward.json to be numeric."""
    values = {"reward": report["reward"]}
    if "invalid_submission" in report:
        return dict(values, invalid_submission=1)
    for key in ("af3_seconds", "clusters_per_allocated_gpu_hour"):
        values[key] = report[key]
    for key in ("seconds_per_slot", "generation_seconds", "total_gpu_seconds", "attempts_per_gpu_hour",
                "generation_attempts_per_gpu_hour"):
        if key in report:
            values[key] = report[key]
    cohorts = [("", report["by_length"])]
    if "comparison" in report:
        ref = report["reference"]
        values.update(report["comparison"], reference_quality=ref["reward"], reference_gpu_seconds=ref["total_gpu_seconds"])
        for key in ("generation_seconds", "af3_seconds", "generation_attempts_per_gpu_hour",
                    "attempts_per_gpu_hour", "clusters_per_allocated_gpu_hour"):
            values[f"reference_{key}"] = ref[key]
        cohorts.append(("reference_", ref["by_length"]))
    for prefix, rows in cohorts:
        for length, row in rows.items():
            for name, value in row.items():
                values[f"{prefix}length_{length}_{name}"] = int(value) if isinstance(value, bool) else value
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
    source = p.add_mutually_exclusive_group()
    source.add_argument("--length", type=int, help="Evaluate one positive monomer length in development")
    source.add_argument("--suite", type=Path, help="Controller-owned JSON request roster")
    p.add_argument("--slots", type=int, help="Override the number of output slots per development request")
    p.add_argument("--reference", choices=NAMES, default="stochastic-tsr", help="Paired executable human baseline")
    p.add_argument("--program-track", action="store_true", help="Adapt legacy policies to the same paired program evaluation")
    p.add_argument("--seconds-per-slot", type=float, default=SECONDS_PER_SLOT,
                   help="Generation/search wall budget; applied to both programs on the same GPU")
    args = p.parse_args()
    if args.length is not None and args.length < 1:
        p.error("--length must be positive")
    if args.slots is not None and args.slots < 1:
        p.error("--slots must be positive")
    if not math.isfinite(args.seconds_per_slot) or args.seconds_per_slot <= 0:
        p.error("--seconds-per-slot must be finite and positive")
    if args.private_seed and (args.slots is not None or args.length is not None):
        p.error("Private evaluation requires the controller's full suite and sample count")
    work, target = args.work.resolve(), args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()) or target.exists():
        p.error("Use a fresh work directory and output path")
    seed = 727000
    final_seed = None
    if args.private_seed:
        seed, final_seed, suite_digest = verifier_runtime().suite_seeds(args.private_seed)
    slots = args.slots or (50 if args.private_seed or args.paper_size else 10)
    try:
        requests = suite_requests(args.suite) if args.suite else None
        lengths = tuple(requests) if requests else ([args.length] if args.length else tuple(LENGTHS))
        package = args.submission
        entry = package / "infer.py"
        if args.program_track and not (entry.exists() or entry.is_symlink()):
            package = package_policy(package, work / "adapted-submission")
        device = gpu_device() if (package / "infer.py").is_file() else None
        submission_sha = package_digest(package) if device else None
        report = run(package, args.assets.resolve(), work, slots, seed, lengths,
                     private=args.private_seed is not None, final_seed=final_seed, seconds_per_slot=args.seconds_per_slot,
                     requests=requests)
        if report["protocol"] == "rfd4-program-v1":
            reference_dir = package_reference(args.reference, work / "reference-package")
            reference = run(reference_dir, args.assets.resolve(), work / "reference", slots, seed, lengths,
                            private=args.private_seed is not None, final_seed=final_seed,
                            seconds_per_slot=args.seconds_per_slot, requests=requests)
            if any(row["program_exit_code"] and not row["timed_out"] for row in reference["by_length"].values()):
                raise RuntimeError(f"The trusted {args.reference} reference failed; inspect its inference.log")
            report.update(reference_name=args.reference, reference=reference, comparison=compare(report, reference),
                          historical_baselines=historical_points(lengths, slots), device=device,
                          submission_sha256=submission_sha, reference_sha256=package_digest(reference_dir))
            # Identify the workload independently of the swept budget.
            cases = []
            for n in lengths:
                case = json.loads((work / str(n) / "request/request.json").read_text())
                case.pop("budget")
                case["input_sha256"] = hashlib.sha256(
                    (work / str(n) / "request" / case["input"]["file"]).read_bytes()).hexdigest()
                cases.append(case)
            report["workload_sha256"] = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
        if args.private_seed:
            report["suite_sha256"] = suite_digest
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

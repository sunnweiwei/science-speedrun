"""Submit complete AWF and batch-preserving parallel LFMC validation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--shard-inputs", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parsable", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", a.name):
        p.error("name must be a plain identifier")
    root, shards, upstream = (
        a.root.resolve(),
        a.shard_inputs.resolve(),
        a.upstream.resolve(),
    )
    track = Path(__file__).resolve().parents[2]
    package = root / "packages" / a.name
    for name in [
        "encoder.pt",
        "config.json",
        "infer.py",
        "provenance.json",
        "LICENSE-OlmoEarth",
    ]:
        if not (package / name).is_file():
            p.error(f"missing exported package file: {name}")
    provenance = json.loads((package / "provenance.json").read_text())
    if provenance.get("training_validity") == "invalid":
        p.error("known invalid training package")
    manifest_path = shards / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        not manifest["full_release"]
        or manifest["start"] != 0
        or manifest["stop"] != manifest["source_rows"]
        or manifest["batch_size"] != 16
        or Path(manifest["source"]).resolve()
        != (root / "eval/lfmc/inputs.h5").resolve()
    ):
        p.error("require full LFMC release and unchanged batch16 boundaries")
    if not manifest["shards"] or not all(
        (shards / item["file"]).is_file() for item in manifest["shards"]
    ):
        p.error("missing input shard")
    receipt = root / "launches" / f"sharded-{a.name}.json"
    if receipt.exists() or any(
        (root / "results" / f"{a.name}-{task}.json").exists()
        or (root / "features" / f"{a.name}-{task}" / "features.npz").exists()
        for task in ["awf", "lfmc"]
    ):
        p.error("existing launch or scores; use a fresh evaluation root")
    context = {
        "gpu_type": "a5000",
        "python_hash_seed": 0,
        "batch_size": 16,
        "namespace_sha256": hashlib.sha256(
            (track / "tests/namespace.py").read_bytes()
        ).hexdigest(),
        "lfmc_shards_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    context_path = root / "execution-context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        p.error("evaluation context differs; use a separate root")
    settings = {
        "OLMO_ROOT": str(root),
        "OLMO_UPSTREAM": str(upstream),
        "OLMO_TRACK": str(track),
        "OLMO_NAME": a.name,
        "OLMO_MODEL": str(package),
        "OLMO_SHARD_INPUTS": str(shards),
        "OLMO_SHARD_OUTPUT": str(root / "sharded-features" / a.name),
        "OLMO_INFERENCE_HASHSEED": "0",
        "OLMO_USE_EXISTING_PACKAGE": "1",
        "OLMO_TASKS": "awf",
    }
    base = [
        "sbatch",
        "--parsable",
        "--partition=" + os.environ.get("EARTH_SLURM_PARTITION", "gpu"),
        "--export=ALL",
    ]
    array = base + [
        "--gres=gpu:1",
        f"--array=0-{len(manifest['shards']) - 1}",
        f"--job-name={a.name}-lfmc-parts",
        f"--output={root}/sharded-%A_%a.log",
        str(track / "docs/slurm/sharded_infer.sbatch"),
    ]
    awf = base + [
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=32G",
        f"--job-name={a.name}-awf",
        f"--output={root}/awf-%j.log",
        str(track / "docs/slurm/validate.sbatch"),
    ]
    merge = base + [
        f"--job-name={a.name}-probes",
        f"--output={root}/merge-%j.log",
        "--dependency=afterok:ARRAY_JOB:AWF_JOB",
        str(track / "docs/slurm/merge_validate.sbatch"),
    ]
    report = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "name": a.name,
        "settings": settings,
        "execution_context": context,
        "array": array,
        "awf": awf,
        "merge": merge,
        "test_metrics_used": False,
    }
    if a.dry_run:
        print(json.dumps(report, indent=2))
        return
    for directory in [root / "results", root / "features", receipt.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    if not context_path.exists():
        with context_path.open("x") as f:
            json.dump(context, f, indent=2)
    with receipt.open("x") as f:
        json.dump(report, f, indent=2)
    env = dict(os.environ, **settings)
    # A CPU dispatcher may have inherited an empty device list. Let each new
    # Slurm GPU allocation set its own device visibility.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for key, command in [("array_job", array), ("awf_job", awf)]:
        output = (
            subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
        )
        if not output.isdigit():
            raise RuntimeError(f"Unexpected Slurm receipt: {output}")
        report[key] = int(output)
        receipt.write_text(json.dumps(report, indent=2) + "\n")
    merge = [
        item.replace("ARRAY_JOB", str(report["array_job"])).replace(
            "AWF_JOB", str(report["awf_job"])
        )
        for item in merge
    ]
    output = subprocess.check_output(merge, env=env, text=True).strip().split(";")[0]
    if not output.isdigit():
        raise RuntimeError(f"Unexpected Slurm receipt: {output}")
    report.update(merge=merge, validation_job=int(output))
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(output if a.parsable else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

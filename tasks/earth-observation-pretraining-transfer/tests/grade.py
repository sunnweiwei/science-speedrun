"""Independent inference, frozen probes, and numeric reward."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

checks = Path(__file__).resolve().parents[1] / "environment/evaluation"
if not checks.is_dir():
    checks = Path(__file__).resolve().parent
if not (checks / "package_contract.py").is_file():
    checks = Path("/evaluation")
sys.path.insert(0, str(checks))
from package_contract import InvalidSubmission, validate_package


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints(model, data, output, probes):
    """Bind deferred scoring to the exact independently replayed artifacts."""
    tests = Path(__file__).resolve().parent
    return {
        "model": {
            name: digest(model / name)
            for name in [
                "encoder.pt",
                "config.json",
                "infer.py",
                "provenance.json",
                "LICENSE-OlmoEarth",
            ]
        },
        "unlabeled_inputs": {
            task: digest(data / task / "inputs.h5") for task in ["awf", "lfmc"]
        },
        "features": {
            task: digest(output / task / "features.npz") for task in ["awf", "lfmc"]
        },
        "verifier": {
            name: digest(tests / name)
            for name in ["grade.py", "replay.py", "namespace.py"]
        },
        "probes": digest(probes),
        "package_contract": digest(checks / "package_contract.py"),
    }


def validate_manifest(manifest, model, data, output, probes):
    if manifest.get("schema_version") != 1 or manifest.get("phase") != "inference":
        raise ValueError("Unsupported independent inference manifest")
    if manifest.get("fingerprints") != fingerprints(model, data, output, probes):
        raise ValueError("Independent replay artifacts or verifier code changed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    phase = p.add_mutually_exclusive_group()
    phase.add_argument(
        "--inference-only",
        action="store_true",
        help="Replay both full input releases without reading labels",
    )
    phase.add_argument(
        "--score-existing",
        action="store_true",
        help="CPU scoring of an exact, manifest-verified prior replay",
    )
    p.add_argument(
        "--probes",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "environment/evaluation/probes.py",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / "inference-manifest.json"
    if (a.output / "reward.json").exists():
        raise ValueError("Scoring already completed; preserve the existing result")
    try:
        evaluate(a, manifest_path)
    except InvalidSubmission as error:
        # Harbor reads reward.json as a flat object of numbers; the reason goes to report.json.
        report = {"reward": 0.0, "status": "invalid_submission", "reason": str(error)}
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (a.output / "reward.json").write_text(json.dumps({"reward": 0.0, "invalid_submission": 1}, indent=2) + "\n")
        print(json.dumps(report))


def evaluate(a, manifest_path):
    validate_package(a.model)
    if a.score_existing:
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, a.model, a.data, a.output, a.probes)
    else:
        if manifest_path.exists() or any(
            (a.output / task / "features.npz").exists() for task in ["awf", "lfmc"]
        ):
            raise ValueError(
                "Replay output already exists; use a fresh output directory"
            )
        for task in ["awf", "lfmc"]:
            out = a.output / task
            out.mkdir(exist_ok=True)
            replay_result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("replay.py")),
                    "--model",
                    str(a.model),
                    "--input",
                    str(a.data / task / "inputs.h5"),
                    "--output",
                    str(out),
                ],
                check=False,
            )
            if replay_result.returncode == 42:
                raise InvalidSubmission("Isolated replay produced invalid features")
            replay_result.check_returncode()
        manifest = {
            "schema_version": 1,
            "phase": "inference",
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "python_hash_seed": os.environ.get("OLMO_INFERENCE_HASHSEED"),
            "fingerprints": fingerprints(a.model, a.data, a.output, a.probes),
            "labels_read": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if a.inference_only:
        print(
            json.dumps(
                {"inference_manifest": str(manifest_path), "test_metrics_used": False}
            )
        )
        return
    scores = {}
    for task in ["awf", "lfmc"]:
        out = a.output / task
        cmd = [
            sys.executable,
            str(a.probes),
            "--features",
            str(out / "features.npz"),
            "--labels",
            str(a.data / task / "labels.json"),
            "--task",
            task,
            "--output",
            str(out / "metrics.json"),
        ]
        if task == "lfmc":
            cmd += ["--final-test"]
        subprocess.run(cmd, check=True)
        scores[task] = json.loads((out / "metrics.json").read_text())
    awf = scores["awf"]["probe"]["validation"]["accuracy"]
    lfmc = scores["lfmc"]["probe"]["test"]["mae"]
    constant = scores["lfmc"]["constant_test"]["mae"]
    skill = max(0.0, 1 - lfmc / max(constant, 1e-12))
    result = {
        "reward": 0.5 * (awf + skill),
        "awf_validation_accuracy": awf,
        "lfmc_test_mae": lfmc,
        "lfmc_test_mae_skill": skill,
    }
    # Harbor reads reward.json as a flat object of numbers; provenance hashes go to report.json.
    report = {
        **result,
        "status": "scored",
        "inference_manifest_sha256": digest(manifest_path),
        "label_file_sha256": {
            task: digest(a.data / task / "labels.json") for task in ["awf", "lfmc"]
        },
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    (a.output / "reward.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

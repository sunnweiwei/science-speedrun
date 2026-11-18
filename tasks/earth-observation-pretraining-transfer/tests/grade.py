"""Independent inference, frozen probes, and V4 spatial numeric reward."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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

AWF_SPATIAL_TEST_MAJORITY_FRACTION = 59.0 / 205.0
LFMC_CONSTANT_TEST_MAE = 24.435180349813233


def default_roster_path():
    env_roster = os.environ.get("OLMO_AWF_KNN_ROSTER", "").strip()
    if env_roster:
        return Path(env_roster)
    return Path(__file__).resolve().parent / "sealed" / "spatial-roster.json"


def default_dev_splits_path():
    for candidate in (
        Path(__file__).resolve().parents[1]
        / "environment"
        / "data"
        / "awf_spatial_dev_splits.json",
        Path("/evaluation/awf_spatial_dev_splits.json"),
        Path("/workspace/data/awf_spatial_dev_splits.json"),
    ):
        if candidate.is_file():
            return candidate
    return None


def compute_earth_v4_score(
    awf_test_accuracy,
    lfmc_test_mae,
    awf_validation_accuracy=None,
    p_maj=AWF_SPATIAL_TEST_MAJORITY_FRACTION,
):
    """Exact V3/V4 Earth Observation score matching harness/ssi_index_v3.py:earth()."""
    a = float(awf_test_accuracy)
    e = float(lfmc_test_mae)
    p = float(p_maj)
    if not (
        math.isfinite(a)
        and math.isfinite(e)
        and math.isfinite(p)
        and 0.0 <= p < 1.0
    ):
        raise ValueError("Nonfinite or invalid Earth V4 score inputs")
    awf_skill = (a - p) / (1.0 - p)
    lfmc_skill = 1.0 - e / LFMC_CONSTANT_TEST_MAE
    signed_score = (awf_skill + lfmc_skill) / 2.0
    score = max(0.0, min(1.0, signed_score))
    payload = {
        "score": score,
        "reward": score,
        "signed_score": signed_score,
        "awf_test_accuracy": a,
        "awf_test_majority_fraction": p,
        "awf_skill": awf_skill,
        "lfmc_test_mae": e,
        "lfmc_skill": lfmc_skill,
        "lfmc_test_mae_skill": lfmc_skill,
    }
    if awf_validation_accuracy is not None:
        payload["awf_validation_accuracy"] = float(awf_validation_accuracy)
    return payload


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
            for name in ["grade.py", "grade_awf_knn.py", "replay.py", "namespace.py"]
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
    p.add_argument(
        "--roster",
        type=Path,
        default=default_roster_path(),
        help="Sealed spatial AWF roster (defaults to tests/sealed/spatial-roster.json)",
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
        report = {
            "score": 0.0,
            "reward": 0.0,
            "status": "invalid_submission",
            "reason": str(error),
        }
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (a.output / "reward.json").write_text(
            json.dumps(
                {"score": 0.0, "reward": 0.0, "invalid_submission": 1}, indent=2
            )
            + "\n"
        )
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

    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import grade_awf_knn

    roster_bytes = a.roster.read_bytes()
    roster = json.loads(roster_bytes)
    awf_labels_bytes = (a.data / "awf" / "labels.json").read_bytes()
    splits = grade_awf_knn.aligned_splits(json.loads(awf_labels_bytes), roster)
    awf = grade_awf_knn.score_features(a.output / "awf" / "features.npz", splits)
    (a.output / "awf" / "metrics.json").write_text(json.dumps(awf, indent=2) + "\n")

    lfmc_out = a.output / "lfmc"
    lfmc_out.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        str(a.probes),
        "--features",
        str(lfmc_out / "features.npz"),
        "--labels",
        str(a.data / "lfmc" / "labels.json"),
        "--task",
        "lfmc",
        "--output",
        str(lfmc_out / "metrics.json"),
        "--final-test",
    ]
    subprocess.run(cmd, check=True)
    lfmc_scores = json.loads((lfmc_out / "metrics.json").read_text())

    lfmc_mae = lfmc_scores["probe"]["test"]["mae"]
    result = compute_earth_v4_score(
        awf_test_accuracy=awf["test"]["accuracy"],
        lfmc_test_mae=lfmc_mae,
        awf_validation_accuracy=awf["valid"]["accuracy"],
    )
    dev_splits_file = default_dev_splits_path()
    # Harbor reads reward.json as a flat object of numbers; provenance hashes and diagnostics go to report.json.
    report = {
        **result,
        "status": "scored",
        "protocol": roster["protocol"] + "/cosine-20nn-v1",
        "roster_source": roster["source"],
        "split_sizes": {name: len(keys) for name, (keys, _) in splits.items()},
        "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "awf_dev_splits_sha256": (
            digest(dev_splits_file) if dev_splits_file is not None else None
        ),
        "awf": awf,
        "lfmc": {
            "selected_parameter": lfmc_scores["probe"]["selected_parameter"],
            "validation": lfmc_scores["probe"]["validation"],
            "test": lfmc_scores["probe"]["test"],
            "constant_test": lfmc_scores["constant_test"],
            "frozen_constant_test_mae": LFMC_CONSTANT_TEST_MAE,
        },
        "inference_manifest_sha256": digest(manifest_path),
        "label_file_sha256": {
            task: digest(a.data / task / "labels.json") for task in ["awf", "lfmc"]
        },
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "reward.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

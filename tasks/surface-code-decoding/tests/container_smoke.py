#!/usr/bin/env python3
"""Exercise actual Docker role separation with tiny synthetic inputs, not accuracy.

Host orchestration uses only Python's standard library. Fixture generation runs
inside the already-built verifier image, whose pinned runtime supplies NumPy.
All three cases use the real supervisor, grader, and Compose topology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]

# An executable without a .sh suffix also tests frozen-artifact execute bits.
DECODER = '''#!/opt/qec/.venv/bin/python
from pathlib import Path
import errno
import json
import os
import socket
import sys
import numpy as np

assert os.getuid() == os.geteuid() == 10001, "inference must run as isolated UID"
assert os.getgid() == os.getegid() == 10001
assert {name for _, name in socket.if_nameindex()} == {"lo"}, "network namespace is not offline"
assert not Path("/private").exists(), "grader labels leaked into inference container"
try:
    list(Path("/sealed").iterdir())
except PermissionError:
    pass
else:
    raise AssertionError("inference can read the trusted sealed volume")
status = Path("/proc/self/status").read_text()
assert any(line.split() == ["NoNewPrivs:", "1"] for line in status.splitlines())
root = Path(__file__).resolve().parents[1]
assert root != Path("/workspace/submission"), "artifact was not relocated"
for target in [root / "model_config.json", Path("/opt/task/qecbench/scoring.py"),
               Path("/benchmark/manifest.json"), Path("/logs/verifier/forged-reward.json"),
               Path("/workspace/submission/forged-file")]:
    try:
        with target.open("ab"):
            pass
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM, errno.EROFS), (target, error)
    else:
        raise AssertionError(f"inference can modify protected path: {target}")
inputs, output = map(Path, sys.argv[1:])
manifest = json.loads((inputs / "manifest.json").read_text())
probability = json.loads((root / "weights/constant.json").read_text())["probability"]
assert probability == 0.1
for condition in manifest["conditions"]:
    with np.load(inputs / condition["arrays"], allow_pickle=False) as source:
        assert set(source.files) == {"sample_ids", "detectors"}, "labels leaked into inference input"
        ids = source["sample_ids"][::-1]
    np.savez(output / (condition["name"] + ".npz"), sample_ids=ids,
             probabilities=np.full(len(ids), probability, dtype=np.float32))
print(json.dumps({"container_security_probe": "passed", "uid": os.getuid(),
                  "conditions": len(manifest["conditions"]), "probability": probability,
                  "network_interfaces": [name for _, name in socket.if_nameindex()],
                  "private_labels_absent": True, "sealed_volume_inaccessible": True,
                  "protected_paths_not_writable": True, "artifact_relocated": True,
                  "extensionless_executable_ran": True}), flush=True)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(root: Path) -> None:
    """Create unmistakably synthetic public inputs and separately held labels."""
    import numpy as np

    sys.path.insert(0, str(PROJECT))
    from baselines.alphaqubit_artifact import EXPECTED_GROUPS
    from qecbench.data import SCHEMA

    inputs, labels = root / "data/verifier", root / "labels"
    rows, target_rows = [], []
    for group in EXPECTED_GROUPS:
        distance, basis = int(group[1]), group[4]
        center = group.split("center_")[1]
        for rounds in range(1, 26, 2):
            name = f"surface_code_b{basis}_d{distance}_r{rounds:02}_center_{center}"
            folder = inputs / "conditions" / name
            folder.mkdir(parents=True)
            ids = np.array([1017, 5, 991, 70], dtype=np.uint64) + np.uint64(len(rows) * 10000)
            packed = np.zeros((4, (rounds * (distance ** 2 - 1) + 7) // 8), dtype=np.uint8)
            np.savez(folder / "arrays.npz", sample_ids=ids, detectors=packed)
            # The constant decoder does not parse circuits; still provide a valid
            # Stim detector-count fixture, never an actual experimental circuit.
            (folder / "circuit_ideal.stim").write_text(
                "# Synthetic container plumbing fixture, not a physical experiment\nR 0\nM 0\n"
                + "DETECTOR rec[-1]\n" * (rounds * (distance ** 2 - 1))
                + "OBSERVABLE_INCLUDE(0) rec[-1]\n"
            )
            label_path = labels / "conditions" / f"{name}.npz"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(label_path, sample_ids=ids, labels=np.array([0, 0, 0, 1], dtype=np.uint8))
            rows.append({"name": name, "group": group, "distance": distance, "basis": basis,
                "rounds": rounds, "shots": 4, "num_detectors": rounds * (distance ** 2 - 1),
                "arrays": f"conditions/{name}/arrays.npz", "arrays_sha256": sha256(folder / "arrays.npz"),
                "circuit": f"conditions/{name}/circuit_ideal.stim", "circuit_sha256": sha256(folder / "circuit_ideal.stim")})
            target_rows.append({"name": name, "shots": 4, "arrays": f"conditions/{name}.npz", "arrays_sha256": sha256(label_path)})
    write_json(inputs / "manifest.json", {"schema": SCHEMA, "split": "test", "synthetic_smoke_only": True,
        "shots": 520, "conditions": list(reversed(rows))})
    write_json(labels / "manifest.json", {"schema": "surface-code-decoding-labels/v1",
        "synthetic_smoke_only": True, "input_manifest_sha256": sha256(inputs / "manifest.json"), "conditions": target_rows})
    submission = root / "submission-valid"
    (submission / "bin").mkdir(parents=True)
    (submission / "bin/decoder").write_text(DECODER)
    (submission / "bin/decoder").chmod(0o755)
    (submission / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd -- "$(dirname -- "$0")" && pwd)"\nexec "$ROOT/bin/decoder" "$1" "$2"\n')
    (submission / "infer.sh").chmod(0o755)
    (submission / "README.md").write_text("Synthetic constant-decoder container smoke. This artifact is not a scientific reference.\n")
    (submission / "requirements.lock").write_text("Uses the verifier image's pinned NumPy runtime; no additional packages.\n")
    write_json(submission / "weights/constant.json", {"probability": 0.1})
    write_json(submission / "model_config.json", {"model": "constant Bernoulli probability", "weights": "weights/constant.json"})
    write_json(submission / "training_provenance.json", {"purpose": "synthetic container security smoke only",
        "fit": "Maximum likelihood on nine synthetic negative and one synthetic positive observation", "scientific_accuracy_evidence": False})
    write_json(submission / "model_manifest.json", {"schema": "surface-code-decoding-model/v1",
        "checkpoints": [{"path": "weights/constant.json", "sha256": sha256(submission / "weights/constant.json")}],
        "model_config": "model_config.json", "training_provenance": "training_provenance.json"})
    shutil.copytree(submission, root / "submission-invalid")
    (root / "submission-invalid/model_manifest.json").write_text("[]\n")
    (root / "missing-labels").mkdir()
    write_json(root / "fixture.json", {"synthetic_smoke_only": True, "conditions": 130, "shots": 520,
        "expected_valid_performance": 0.75, "source": "deterministic synthetic fixture; no downloaded scientific data"})


def command(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
            timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(args), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    return result


def collect_reports(container: str, destination: Path) -> None:
    """Copy individual evidence files without exporting the protected artifact.

    Recursively copying /logs/verifier also exports the frozen 0555 submission
    tree. Docker then cannot create its children as the unprivileged host user.
    Keep those security permissions intact and request only reviewable files.
    """
    for relative in ("report.json", "reward.json", "inference/inference.log"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        result = command(["docker", "cp", f"{container}:/logs/verifier/{relative}", str(target)], check=False)
        if result.returncode:
            # An infrastructure failure must have no reward; an invalid
            # manifest has no inference log. Other copy errors remain errors.
            missing = "Could not find the file" in result.stdout
            if relative != "report.json" and missing:
                continue
            raise RuntimeError(f"Cannot collect {relative} from {container}: {result.stdout.strip()}")


def case_run(case: str, fixture: Path, output: Path, image: str) -> dict[str, Any]:
    """Run a fresh main/grader pair and private sealed volume per scenario."""
    project = "qec-smoke-" + uuid.uuid4().hex[:12]
    container = project + "-inference"
    case_output = output / case
    case_output.mkdir(parents=True)
    submission = fixture / ("submission-invalid" if case == "invalid-submission" else "submission-valid")
    labels = fixture / ("missing-labels" if case == "missing-labels" else "labels")
    override = fixture / f"override-{case}.json"
    write_json(override, {"services": {
        "main": {"image": image, "cpus": 2, "mem_limit": "4g", "shm_size": "256m",
            "volumes": [{"type": "bind", "source": str(submission), "target": "/workspace/submission",
                         "read_only": True, "bind": {"create_host_path": False}}]},
        "grader": {"image": image, "cpus": 2, "mem_limit": "4g",
                   "command": ["python", "/opt/task/tests/grader.py", "--wait-seconds", "180"]}}})
    env = dict(os.environ, TASK_DATA_DIR=str(fixture / "data"), TASK_PRIVATE_DIR=str(labels))
    compose = ["docker", "compose", "--project-name", project, "-f", str(TASK / "tests/docker-compose.yaml"),
               "-f", str(TASK / "tests/docker-compose.cpu-smoke.yaml"), "-f", str(override)]
    # Check the actual Compose merge: CI must not silently request a GPU.
    resolved = json.loads(command(compose + ["config", "--format", "json"], env=env).stdout)
    for service in resolved["services"].values():
        assert not service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
        assert service["network_mode"] == "none"
    run: subprocess.CompletedProcess[str] | None = None
    try:
        command(compose + ["up", "-d", "--no-build", "grader"], env=env)
        run = command(compose + ["run", "--no-deps", "--name", container, "-e", "QEC_INFERENCE_TIMEOUT_SECONDS=60",
                                  "main", "/opt/task/tests/test.sh"], env=env, timeout=420, check=False)
        (case_output / "supervisor.log").write_text(run.stdout)
        collect_reports(container, case_output)
        grader_logs = command(compose + ["logs", "--no-color", "grader"], env=env, check=False)
        (case_output / "grader.log").write_text(grader_logs.stdout)
        report_path = case_output / "report.json"
        if not report_path.is_file():
            raise AssertionError(f"{case}: supervisor did not publish a report; see {case_output}")
        report = json.loads(report_path.read_text())
        reward_path = case_output / "reward.json"
        config = json.loads((PROJECT / "score_config.json").read_text())
        if case == "valid":
            assert run.returncode == 0, run.stdout
            assert report["status"] == "valid" and report["model_quality_measured"] is True
            assert report["shots"] == 520 and report["conditions"] == 130
            assert report["performance"] == 0.75 and report["balanced_logical_failure"] == 0.25
            expected = max(0.0, (0.75 - config["lower_bound"]) / (config["upper_bound"] - config["lower_bound"]))
            assert math.isclose(report["score"], expected, rel_tol=0, abs_tol=1e-12)
            assert json.loads(reward_path.read_text())["reward"] == report["performance"]
            inference_log = (case_output / "inference/inference.log").read_text()
            probes = [json.loads(line) for line in inference_log.splitlines() if line.startswith('{"container_security_probe"')]
            assert len(probes) == 1 and probes[0]["container_security_probe"] == "passed"
        elif case == "invalid-submission":
            assert run.returncode == 0 and report["status"] == "invalid_submission"
            assert report["model_quality_measured"] is False and report["performance"] is None
            assert json.loads(reward_path.read_text()) == {"reward": 0.0}
        else:
            assert run.returncode != 0 and report["status"] == "infrastructure_error"
            assert report["model_quality_measured"] is False and not reward_path.exists()
        return {"case": case, "status": "passed", "verifier_status": report["status"],
                "supervisor_exit_code": run.returncode, "score": report.get("score"),
                "scientific_accuracy_evidence": False}
    finally:
        command(["docker", "rm", "-f", container], check=False)
        command(compose + ["down", "--volumes", "--remove-orphans"], env=env, check=False)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    command(["docker", "version"])
    command(["docker", "compose", "version"])
    if args.build:
        command(["docker", "build", "-t", args.image, "-f", str(TASK / "tests/Dockerfile"), str(TASK)], timeout=2400)
    with tempfile.TemporaryDirectory(prefix="qec-container-smoke-") as temporary:
        fixture = Path(temporary)
        command(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                 "--mount", f"type=bind,src={fixture},dst=/fixture", args.image,
                 "python", "/opt/task/tests/container_smoke.py", "--prepare", "/fixture"])
        results = [case_run(case, fixture, args.output, args.image)
                   for case in ("valid", "invalid-submission", "missing-labels")]
    write_json(args.output / "container-smoke.json", {"schema": "surface-code-container-smoke/v1",
        "status": "passed", "scientific_accuracy_evidence": False,
        "scope": "Actual native Docker CPU supervisor/grader role separation and error classification on synthetic data",
        "cases": results, "wall_seconds": time.monotonic() - started})
    print(f"Container smoke passed; evidence: {args.output / 'container-smoke.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="Internal: generate synthetic fixtures using the image's NumPy")
    parser.add_argument("--image", default="surface-code-verifier:smoke")
    parser.add_argument("--build", action="store_true", help="Build the pinned verifier image before the smoke")
    parser.add_argument("--output", type=Path, default=Path("container-smoke-results"))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        args.output = args.output.resolve()
        orchestrate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Native Docker deployment checks on synthetic data, not accuracy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
import uuid

TASK = Path(__file__).resolve().parents[1]
INFER = r'''import json, os, socket, sys
from pathlib import Path
import torch

source, output = map(Path, sys.argv[1:])
assert os.getuid() == 10001
assert {name for _, name in socket.if_nameindex()} == {"lo"}
assert not Path("/private").exists()
try:
    list(Path("/sealed").iterdir())
except PermissionError:
    pass
else:
    raise AssertionError("inference can access the sealed volume")
for name in ("/tests/grader.py", "/tests/score_bounds.json", "/benchmark/manifest.json",
             "/workspace/submission/model_manifest.json", "/logs/verifier/reward.json"):
    try:
        with open(name, "ab"):
            pass
    except (PermissionError, OSError):
        pass
    else:
        raise AssertionError("protected path is writable: " + name)
queries = [json.loads(line) for line in (source / "queries.jsonl").read_text().splitlines()]
assert all("true_smiles" not in row and "correct_candidate_id" not in row for row in queries)
pools = {row["candidate_pool_id"]: row["candidates"] for row in
         map(json.loads, (source / "candidate_pools.jsonl").read_text().splitlines())}
state = torch.load(Path(__file__).with_name("model.pt"), map_location="cpu", weights_only=True)
config = json.loads(Path(__file__).with_name("config.json").read_text())
with (output / "rankings.jsonl").open("w") as stream:
    for row in reversed(queries):
        ranking = [pools[row["candidate_pool_id"]][i]["candidate_id"] for i in state["order"].tolist()]
        if config["malformed_output"]:
            ranking = ranking[:1]
        stream.write(json.dumps({"query_id": row["query_id"], "ranking": ranking}) + "\n")
print(json.dumps({"container_security_probe": "passed", "uid": os.getuid(),
                  "queries": len(queries), "network_interfaces": ["lo"],
                  "private_labels_absent": True, "sealed_volume_inaccessible": True,
                  "protected_paths_not_writable": True}), flush=True)
'''


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path) -> None:
    """Generate four queries and a real, tiny PyTorch checkpoint inside the image."""
    import torch

    inputs, labels, submission = root / "data/verifier", root / "labels", root / "submission-valid"
    for directory in (inputs, labels, submission, root / "missing-labels"):
        directory.mkdir(parents=True)
    candidates = [{"candidate_id": hashlib.sha256(s.encode()).hexdigest(), "smiles": s} for s in ("C", "CC")]
    queries = [{"query_id": f"synthetic-{i}", "mzs": [10.0], "intensities": [1.0],
                "precursor_mz": 20.0, "adduct": "[M+H]+", "instrument_type": "synthetic",
                "collision_energy": None, "candidate_pool_id": "synthetic-pool"} for i in range(4)]
    (inputs / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in queries))
    (inputs / "candidate_pools.jsonl").write_text(json.dumps({"candidate_pool_id": "synthetic-pool", "candidates": candidates}) + "\n")
    write_json(inputs / "manifest.json", {"schema": "massspecgym-retrieval-input/v1", "synthetic_smoke_only": True,
        "counts": {"queries": 4, "candidate_pools": 1},
        "files": {name: digest(inputs / name) for name in ("queries.jsonl", "candidate_pools.jsonl")}})
    truths = [{"query_id": row["query_id"], "correct_candidate_id": candidates[int(i > 0)]["candidate_id"],
               "true_smiles": candidates[int(i > 0)]["smiles"]} for i, row in enumerate(queries)]
    (labels / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in truths))
    write_json(labels / "manifest.json", {"schema": "massspecgym-retrieval-private-labels/v1",
        "synthetic_smoke_only": True, "files": {"labels.jsonl": digest(labels / "labels.jsonl")}})
    torch.save({"order": torch.tensor([0, 1])}, submission / "model.pt")
    write_json(submission / "config.json", {"malformed_output": False})
    (submission / "infer.py").write_text(INFER)
    (submission / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd -- "$(dirname -- "$0")" && pwd)"\nexec python "$ROOT/infer.py" "$1" "$2"\n')
    (submission / "README.md").write_text("Synthetic deployment fixture only; no scientific model or accuracy claim.\n")
    write_json(submission / "model_manifest.json", {"schema": "massspecgym-retrieval-model/v1",
        "checkpoint": {"path": "model.pt", "sha256": digest(submission / "model.pt"), "format": "pytorch"},
        "model_config": "config.json", "training_provenance": {
            "data": "Synthetic constant ordering fixture, not a trained scientific baseline", "pretrained_assets": []}})
    for case in ("invalid-output", "invalid-submission"):
        shutil.copytree(submission, root / f"submission-{case}")
    write_json(root / "submission-invalid-output/config.json", {"malformed_output": True})
    write_json(root / "submission-invalid-submission/model_manifest.json", {})


def command(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 600,
            check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(args), flush=True)
    result = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    print(result.stdout, end="", flush=True)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    return result


def case_run(case: str, fixture: Path, output: Path, image: str) -> dict[str, Any]:
    project = "massspec-smoke-" + uuid.uuid4().hex[:12]
    container, evidence = project + "-inference", output / case
    evidence.mkdir(parents=True)
    submission = fixture / ("submission-" + case if case.startswith("invalid-") else "submission-valid")
    env = dict(os.environ, TASK_DATA_DIR=str(fixture / "data"),
               TASK_PRIVATE_DIR=str(fixture / ("missing-labels" if case == "missing-labels" else "labels")),
               MASSSPECGYM_SUBMISSION_DIR=str(submission))
    override = fixture / f"override-{case}.yaml"
    # Clear GPU requests from the real local evaluation profile. !reset requires
    # Compose >= 2.30 and prevents accidental GPU-dependent CPU smoke tests.
    override.write_text(f'''services:
  main:
    image: {json.dumps(image)}
    cpus: 2
    mem_limit: 4g
    shm_size: 256m
    gpus: !reset []
    deploy:
      resources:
        reservations:
          devices: !reset []
  grader:
    image: {json.dumps(image)}
    cpus: 2
    mem_limit: 4g
    command: [python, /tests/grader.py, --sealed, /sealed/run, --input, /benchmark, --labels, /private, --output, /sealed/results, --wait-seconds, '180', --mces-workers, '1']
''')
    base = ["docker", "compose", "--project-name", project, "-f", str(TASK / "tests/docker-compose.yaml"),
            "-f", str(TASK / "tests/docker-compose.local.yaml")]
    compose = base + ["-f", str(override)]
    try:
        profile = json.loads(command(base + ["config", "--format", "json"], env=env).stdout)
        devices = profile["services"]["main"]["deploy"]["resources"]["reservations"]["devices"]
        assert len(devices) == 1 and devices[0]["count"] == 1 and devices[0]["capabilities"] == ["gpu"]
        cpu = json.loads(command(compose + ["config", "--format", "json"], env=env).stdout)
        for service in cpu["services"].values():
            assert service["network_mode"] == "none" and not service.get("gpus")
            assert not service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
        write_json(evidence / "compose.json", cpu)
        command(compose + ["up", "-d", "--no-build", "grader"], env=env)
        result = command(compose + ["run", "--no-deps", "--name", container, "-e",
                                   "MASSSPECGYM_INFERENCE_TIMEOUT_SECONDS=60", "main", "/tests/test.sh"],
                         env=env, timeout=300, check=False)
        (evidence / "supervisor.log").write_text(result.stdout)
        for relative in ("report.json", "reward.json", "main/inference.log"):
            target = evidence / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = command(["docker", "cp", f"{container}:/logs/verifier/{relative}", str(target)], check=False)
            if copied.returncode and not (relative != "report.json" and "Could not find the file" in copied.stdout):
                raise RuntimeError(f"Failed to collect {relative}: {copied.stdout}")
        (evidence / "grader.log").write_text(command(compose + ["logs", "--no-color", "grader"], env=env).stdout)
        report = json.loads((evidence / "report.json").read_text())
        reward = evidence / "reward.json"
        if case == "missing-labels":
            assert result.returncode != 0 and report["status"] == "infrastructure_error" and not reward.exists()
        else:
            assert result.returncode == 0 and report["status"] == {"valid": "complete", "invalid-output": "invalid_output", "invalid-submission": "invalid_submission"}[case]
            hit = 0.25 if case == "valid" else 0.0
            expected = hit   # the reward is the raw Hit@1
            assert report["queries"] == 4 and report["metrics"]["hit_rate_at_1"] == hit
            assert math.isclose(json.loads(reward.read_text())["reward"], expected, rel_tol=0, abs_tol=1e-12)
            assert report["model_quality_measured"] == (case == "valid")
            if case == "valid":
                assert report["metrics"]["hit_rate_at_5"] == 1.0 and math.isfinite(report["metrics"]["mces_at_1"])
                assert '"container_security_probe": "passed"' in (evidence / "main/inference.log").read_text()
        return {"case": case, "status": "passed", "verifier_status": report["status"]}
    finally:
        command(["docker", "rm", "-f", container], check=False)
        command(compose + ["down", "--volumes", "--remove-orphans"], env=env, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--image", default="massspec-verifier:smoke")
    parser.add_argument("--development-image", default="massspec-development:smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    if args.output is None:
        parser.error("--output is required unless --prepare is used")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    command(["docker", "version"])
    command(["docker", "compose", "version"])
    # A missing real release must fail closed. No fake full-development manifest
    # is constructed to bypass the production resource completeness checks.
    development = command(["docker", "run", "--rm", "--network", "none", args.development_image, "true"], check=False)
    (args.output / "development.log").write_text(development.stdout)
    assert development.returncode == 66 and "development manifest is missing" in development.stdout
    with tempfile.TemporaryDirectory(prefix="massspec-container-smoke-") as temporary:
        fixture = Path(temporary)
        command(["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                 "--mount", f"type=bind,src={fixture},dst=/fixture", args.image,
                 "python", "/tests/container_smoke.py", "--prepare", "/fixture"])
        cases = [case_run(case, fixture, args.output, args.image)
                 for case in ("valid", "invalid-output", "invalid-submission", "missing-labels")]
    write_json(args.output / "container-smoke.json", {"schema": "massspec-container-smoke/v1", "status": "passed",
        "scope": "Native Docker CPU deployment checks with four synthetic queries; not HF materialization or scientific reproduction",
        "scientific_accuracy_evidence": False, "development_missing_data_fails_closed": True,
        "cases": cases, "wall_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

"""Trusted release preflight, relocated replay, and known-task score aggregation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from protocol import InfrastructureError, PolicyError, numbers
from native_settings import validate_settings

# The verifier image owns this sibling data specification; candidate code does
# not choose the benchmark membership or mutate its inclusion/exclusion list.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from public_release import load_release, digest as data_release_hash, validate_suite_binding


SOURCE_REVISION = "cf5ef6748443ae72e9647f398487e98d421a7243"


def write_json(path, value):
    # External readers can inspect progress without observing half-written JSON.
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def suite_hash(suite):
    return hashlib.sha256(json.dumps(suite, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_suite(suite, *, require_reference=True):
    if suite.get("status") != "validated" or suite.get("coverage", {}).get("status") != "complete":
        raise InfrastructureError("Evaluation release is not validated/full-coverage; do not run a formal agent session")
    if not suite.get("release_id") or suite.get("source_revision") != SOURCE_REVISION:
        raise InfrastructureError("Missing release identity or wrong upstream resource revision")
    if not suite.get("cases"):
        raise InfrastructureError("Empty evaluation suite")
    timeout = suite.get("policy_timeout_sec", 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise InfrastructureError("Invalid policy timeout")
    ids = set()
    for case in suite["cases"]:
        for field in ("id", "environment", "family"):
            if not isinstance(case.get(field), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case[field]):
                raise InfrastructureError(f"Unsafe/missing case {field}")
        if case["id"] in ids:
            raise InfrastructureError("Duplicate evaluation case")
        ids.add(case["id"])
        if case.get("backend") != "maia":
            raise InfrastructureError(f"Backend not integrated: {case.get('backend')}; never silently skip a case")
        for key in ("horizon", "solver_processes", "observation_size", "action_size", "timeout_sec"):
            if type(case.get(key)) is not int or case[key] <= 0:
                raise InfrastructureError(f"Invalid {key} in {case['id']}")
        validate_settings(case.get("native_settings", {}), case["horizon"],
                          case.get("episode_end", "native"))
        seeds = case.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise InfrastructureError("Invalid evaluation seeds")
        if not case.get("initialization") or not case.get("provenance"):
            raise InfrastructureError("Missing initialization/protocol provenance")
        kwargs = case.get("env_kwargs", {})
        allowed = {"probe_locations", "obs_normalization_strategy", "obs_loc", "obs_scale"}
        if set(kwargs) - allowed or not kwargs.get("probe_locations") or not numbers(kwargs["probe_locations"]):
            raise InfrastructureError("Missing reviewed sensors or unsafe environment overrides")
        strategy = kwargs.get("obs_normalization_strategy", "none")
        if strategy not in ("none", "U_inf", "customized"):
            raise InfrastructureError("Evaluation normalization must be fixed, never fitted during replay")
        if strategy == "customized":
            for key in ("obs_loc", "obs_scale"):
                if not numbers(kwargs.get(key)) or len(kwargs[key]) != case["observation_size"]:
                    raise InfrastructureError("Missing/invalid frozen normalization statistics")
            if any(x <= 0 for x in kwargs["obs_scale"]):
                raise InfrastructureError("Frozen normalization scale must be positive")
        if not require_reference:
            continue
        baseline = case.get("baseline", {})
        anchor, reference = baseline.get("anchor_return"), baseline.get("reference_return")
        if not isinstance(baseline.get("anchor_description"), str) or not baseline["anchor_description"].strip():
            raise InfrastructureError("Missing native uncontrolled/fixed-control anchor definition")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (anchor, reference)) or reference <= anchor:
            raise InfrastructureError("Missing measured reference anchors or nonpositive normalization gap")
        if baseline.get("algorithm") not in ("PPO", "DDPG", "TD3", "GPPO"):
            raise InfrastructureError("Reference algorithm must be a reproduced native baseline")


def preflight(data, *, require_reference=True):
    try:
        suite = json.loads((data / "suite.json").read_text())
        release_path = data / "release.json"
        # A missing manifest must not reopen legacy, arbitrary-subset grading.
        release = load_release(release_path)
        if release != load_release():
            raise InfrastructureError("Mounted data release differs from the verifier's frozen selection")
        validate_suite_binding(release, suite)
        validate_suite(suite, require_reference=require_reference)
        receipt = json.loads((data / "materialized.json").read_text())
        if receipt.get("revision") != SOURCE_REVISION:
            raise InfrastructureError("Resources have not been verified at the pinned revision")
        if (receipt.get("data_release_id") != release["release_id"] or
                receipt.get("data_release_sha256") != data_release_hash(release)):
            raise InfrastructureError("Frozen release resources have not been materialized/verified")
        verified = set(receipt.get("verified_directories", []))
        for case in suite["cases"]:
            directory = data / "environments" / case["environment"]
            if case["environment"] not in verified or not (directory / "environment_config.yaml").is_file():
                raise InfrastructureError(f"Unverified/missing resource: {case['environment']}")
        return suite
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise InfrastructureError(f"Incomplete evaluation release at {data}: {error}") from error


def package_hash(root):
    root = root.resolve()
    if not root.is_dir():
        raise PolicyError("No submission directory")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.resolve().is_relative_to(root):
            raise PolicyError("Submission contains an escaping symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PolicyError("Submission contains a missing link or special file")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    for required in ("infer.sh", "README.md"):
        if not (root / required).is_file() or not (root / required).stat().st_size:
            raise PolicyError(f"Missing submission {required}")
    return digest.hexdigest()


def aggregate(suite, results, *, normalize=True):
    expected = {case["id"] for case in suite["cases"]}
    if set(results) != expected:
        raise InfrastructureError("Incomplete/extra result coverage")
    scores = []
    valid = 0
    for case in suite["cases"]:
        result = results[case["id"]]
        if result.get("status") == "policy_error":
            if normalize:
                result["score"] = 0.0
                scores.append(0.0)
            else:
                result["mean_return"] = None
            continue
        if result.get("status") != "ok":
            raise InfrastructureError("A solver/setup error is not a model score")
        episodes = result.get("episodes", [])
        if [e.get("seed") for e in episodes] != case["seeds"]:
            raise InfrastructureError("Wrong/missing evaluation episodes")
        values = [e["return"] for e in episodes]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise InfrastructureError("Nonfinite/missing native return")
        mean = sum(values) / len(values)
        if not math.isfinite(mean):
            raise InfrastructureError("Nonfinite mean return")
        result["mean_return"] = mean
        valid += 1
        if not normalize:
            continue
        anchors = case["baseline"]
        gain = (mean - anchors["anchor_return"]) / (anchors["reference_return"] - anchors["anchor_return"])
        if not math.isfinite(gain):
            raise InfrastructureError("Nonfinite normalized return")
        result.update(mean_return=mean, normalized_gain=gain, score=max(0.0, gain))
        scores.append(result["score"])
    if not normalize:
        return {"case_coverage": valid / len(suite["cases"])}
    return {"score": sum(scores) / len(scores), "case_coverage": valid / len(scores)}


def run_case(case, suite, data, package, output):
    command = [sys.executable, str(Path(__file__).with_name("run_maia.py")), "evaluate",
               "--data", str(data), "--case", case["id"], "--submission", str(package),
               "--output", str(output)]
    with output.with_suffix(".log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=case["timeout_sec"])
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(f"CFD case timeout: {case['id']}") from error
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    if code or not output.is_file():
        raise InfrastructureError(f"CFD case execution failed: {case['id']}; see {output.with_suffix('.log')}")
    return json.loads(output.read_text())


def evaluate(data, submission, output, runner=run_case, *, raw_only=False):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise InfrastructureError("Evaluation output directory must be empty")
    try:
        suite = preflight(data, require_reference=not raw_only)
        release_digest = suite_hash(suite)
        identity = {"release_id": suite["release_id"], "source_revision": SOURCE_REVISION,
                    "suite_sha256": release_digest,
                    "evaluation_mode": "native-returns" if raw_only else "normalized-score"}
        try:
            digest = package_hash(submission)
        except PolicyError as error:
            write_json(output / "results.json", dict(identity, status="policy_error", error=str(error),
                       case_coverage=0.0, cases={case["id"]: {"status": "policy_error", "error": str(error)}
                                                 for case in suite["cases"]}))
            if not raw_only:
                write_json(output / "reward.json", {"score": 0.0, "case_coverage": 0.0})
            return
        with tempfile.TemporaryDirectory(prefix="hydrogym-replay-") as temporary:
            package = Path(temporary) / "submission"
            shutil.copytree(submission, package, symlinks=True)
            if package_hash(package) != digest or package_hash(submission) != digest:
                raise InfrastructureError("Submission changed while snapshotting; retry a stable version")
            results = {}
            for case in suite["cases"]:
                results[case["id"]] = runner(case, suite, data, package, output / f"{case['id']}.json")
                if package_hash(package) != digest:
                    raise InfrastructureError("Model package mutated during replay")
            if suite_hash(preflight(data, require_reference=not raw_only)) != release_digest:
                raise InfrastructureError("Evaluation release changed during replay")
            rewards = aggregate(suite, results, normalize=not raw_only)
            write_json(output / "results.json", dict(identity, submission_sha256=digest,
                       status="ok", case_coverage=rewards["case_coverage"], cases=results))
            if not raw_only:
                write_json(output / "reward.json", rewards)
    except Exception as error:
        write_json(output / "error.json", {"status": "infrastructure_error", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--raw-only", action="store_true",
                        help="Evaluate every declared case without reference anchors; no aggregate reward file")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.data.resolve(), require_reference=not args.raw_only), indent=2))
    else:
        evaluate(args.data.resolve(), args.submission.resolve(), args.output.resolve(), raw_only=args.raw_only)


if __name__ == "__main__":
    main()

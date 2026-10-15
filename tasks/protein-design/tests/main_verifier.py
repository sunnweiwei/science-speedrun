#!/usr/bin/env python3
"""Run the untrusted submission offline and seal only bounded design output.

The submission generates backbones; it never sees the scoring pipeline. Sealing copies
the design index and the referenced structures into a read-only, hash-manifested
snapshot that the trusted grader consumes.
"""

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import subprocess
import time

from common import (SNAPSHOT_SCHEMA, TRACKS, sha256, validate_harness,
                    validate_input, write_json)

MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_STRUCTURES = 100_000



def invalid_submission(output, sealed, error):
    """An invalid package, a failed inference or a timeout scores 0 (TASK_STANDARD.md); Harbor needs reward.json."""
    reason = f"{type(error).__name__}: {error}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(
        {"status": "invalid_submission", "error": reason[:4000], "reward": 0.0}, indent=2) + "\n")
    (output / "reward.json").write_text(json.dumps({"reward": 0.0, "invalid_submission": 1.0}, indent=2) + "\n")
    sealed.parent.mkdir(parents=True, exist_ok=True)
    (sealed.parent / "INVALID").write_text(reason + "\n")
    print(f"invalid submission: {reason}")

def validate_submission(root):
    root = root.resolve()
    for required in ("infer.sh", "README.md", "harness.json"):
        if not (root / required).is_file() or (root / required).is_symlink():
            raise FileNotFoundError(f"submission/{required} is required")
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")


def check_grading_instruments(root, manifest):
    """Record when a submission declares the models that grade it.

    Pretrained predictors are permitted - a `predictor_inversion` system is nothing but
    a pretrained predictor - so this records rather than rejects. It matters because a
    system optimising against the very models that score it inflates in-silico
    designability, and a reviewer should see it.

    This is declaration-based. It matters MORE now than when AlphaFold 3 was the
    predictor: AF3's parameters could not be redistributed, so the scorer was never in
    the agent's image and inverting it needed the agent to bring its own copy. RF3's
    parameters DO ship, so the scoring predictor is sitting in the image the agent runs
    in, and optimising directly against it is correspondingly easier. The declaration
    is still the only signal - this does not scan bytes.
    """
    from pipeline import RF3_REVISION, LIGANDMPNN_COMMIT  # noqa: F401
    instruments = ("ligandmpnn", "ligand_mpnn", "proteinmpnn", "protein_mpnn",
                   "rosettafold3", "rosettafold-3", "rf3", "foundry",
                   "alphafold3", "alphafold-3", "af3")
    found = []
    for asset in manifest.get("pretrained_assets", []):
        haystack = f"{asset.get('asset_id', '')} {asset.get('role', '')}".lower()
        for name in instruments:
            if name in haystack:
                found.append({"asset": name, "declared_as": asset.get("asset_id")})
                break
    return found


def kill_agent_processes(uid):
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            status = (entry / "status").read_text()
            process_uid = int(next(line for line in status.splitlines()
                                   if line.startswith("Uid:")).split()[1])
            if process_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass


def run_submission(submission, input_root, work, timeout):
    agent = pwd.getpwnam("agent")
    output = work / "untrusted-output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    env = {
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": agent.pw_dir, "PYTHONUNBUFFERED": "1", "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }

    def demote():
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)

    process = None
    try:
        with (work / "inference.log").open("wb") as log:
            process = subprocess.Popen(
                ["bash", str(submission / "infer.sh"), str(input_root), str(output)],
                cwd=submission, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=demote,
            )
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    except subprocess.TimeoutExpired:
        if process:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
        kill_agent_processes(agent.pw_uid)
    return output


def validate_output_tree(root):
    total, count = 0, 0
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            info = (Path(directory) / name).lstat()
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode)
                                                  or stat.S_ISREG(info.st_mode)):
                raise ValueError("design output contains a symlink or special file")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                count += 1
                if total > MAX_OUTPUT_BYTES:
                    raise ValueError("design output exceeds 2 GiB")
                if count > MAX_STRUCTURES + len(TRACKS):
                    raise ValueError("design output contains too many files")


def seal_designs(untrusted, sealed, input_manifest):
    if sealed.exists():
        raise FileExistsError("sealed destination already exists")
    validate_output_tree(untrusted)
    predictions = untrusted / "predictions"
    temporary = sealed.with_name(sealed.name + f".tmp-{os.getpid()}")
    (temporary / "structures").mkdir(parents=True)
    hashes, structures = {}, 0

    for track in TRACKS:
        source = predictions / f"{track}.jsonl"
        if not source.is_file() or source.is_symlink():
            continue
        if not source.resolve().is_relative_to(untrusted.resolve()):
            raise ValueError("design index escapes the output root")
        destination = temporary / f"{track}.jsonl"
        shutil.copyfile(source, destination, follow_symlinks=False)
        hashes[destination.name] = sha256(destination)
        destination.chmod(0o444)

    source_structures = predictions / "structures"
    if source_structures.is_dir() and not source_structures.is_symlink():
        for path in sorted(source_structures.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            if not path.resolve().is_relative_to(source_structures.resolve()):
                raise ValueError("structure path escapes the output root")
            relative = path.relative_to(source_structures)
            target = temporary / "structures" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target, follow_symlinks=False)
            target.chmod(0o444)
            structures += 1

    snapshot = {"schema": SNAPSHOT_SCHEMA, "input_manifest_sha256": sha256(input_manifest),
                "files": hashes, "structures": structures}
    write_json(temporary / "snapshot.json", snapshot)
    (temporary / "snapshot.json").chmod(0o444)
    os.replace(temporary, sealed)
    (sealed.parent / "READY").write_text(sha256(sealed / "snapshot.json") + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--input", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--pretrained", default="/pretrained", type=lambda x: Path(x).resolve())
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--timeout", type=int, default=43200)
    # How long to wait for the trusted grader AFTER inference. Must match the grader's
    # own patience (`grader.py --wait-seconds`) or a correct run is killed for being slow.
    # The evaluation view is 51 targets x 20 designs = 1,020, and grading folds ~8,600
    # structures (the 100-design view took 9 h 33 min on 8x L40S). Together with --timeout this must stay
    # within `task.toml [verifier] timeout_sec`. This was hard-coded to 3600 and aborted
    # every honest full-path run at the one-hour mark with "trusted grader did not publish
    # reward.json".
    parser.add_argument("--grader-timeout", type=int, default=129600)
    parser.add_argument("--no-wait", action="store_true")
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    os.chmod(arguments.output, 0o777)
    arguments.sealed.parent.mkdir(parents=True, exist_ok=True)
    validate_input(arguments.input)
    try:
        validate_submission(arguments.submission)
        manifest = validate_harness(arguments.submission, arguments.pretrained)
        instruments = check_grading_instruments(arguments.submission, manifest)
        work = arguments.output / "main"
        work.mkdir()
        write_json(arguments.output / "harness.json", {
            "kind": manifest["kind"],
            "declared_parameters": manifest["declared_parameters"],
            "trained_in_session": [c["path"] for c in manifest["learned_components"]
                                   if c["trained_in_session"]],
            "pretrained_assets": [a["asset_id"] for a in manifest["pretrained_assets"]],
            "grading_instruments_present": instruments,
        })
        untrusted = run_submission(arguments.submission, arguments.input, work, arguments.timeout)
        seal_designs(untrusted, arguments.sealed, arguments.input / "manifest.json")
    except (FileNotFoundError, ValueError, subprocess.SubprocessError) as error:
        invalid_submission(arguments.output, arguments.sealed, error)
        return

    if not arguments.no_wait:
        results = arguments.sealed.parent / "results"
        deadline = time.monotonic() + arguments.grader_timeout
        while not (results / "reward.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"trusted grader did not publish reward.json within "
                    f"{arguments.grader_timeout}s")
            time.sleep(1)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(results / name, arguments.output / name)


if __name__ == "__main__":
    main()

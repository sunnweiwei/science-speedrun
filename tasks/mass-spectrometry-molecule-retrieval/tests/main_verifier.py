#!/usr/bin/env python3
"""Replay an untrusted model on input-only data and seal its ranking output."""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import pwd
import random
import selectors
import shutil
import signal
import stat
import subprocess
import time
import sys

# In the verifier image these shared files are copied beside this module.
# In a checkout import the same source used by development validation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/evaluation"))
from benchmark_score import normalize

from common import (
    REPLAY_PROTOCOL, SPOT_CHECK_MAX_QUERIES, STARTUP_SECONDS_PER_INVOCATION,
    STREAM_PROTOCOL, VERIFIER_BASE_SECONDS, compact_json, contained, sha256,
    validate_input, write_json,
)


# On the frozen test each candidate pool corresponds to one exact target SMILES.
# MassSpecGym ranks every spectrum on its own, so no invocation may see two spectra of one pool.
BATCH_ORDER_SEED = "massspecgym-retrieval-batches/v1"
STREAM_PRIVATE_SEED = "massspecgym-retrieval-stream-spotcheck-v4/9f4c2e71b80d43a6"
MAX_STREAM_LINE_BYTES = 4 * 1024 * 1024
INPUT_FILES = ("queries.jsonl", "candidate_pools.jsonl")
# Every invocation reloads the model, so the inference budget is the compute timeout (18,000 s)
# plus this start-up allowance per invocation, and the verifier budget grows by the same amount.


def inference_budget(timeout: float, invocations: int) -> float:
    return timeout + STARTUP_SECONDS_PER_INVOCATION * invocations


class InvalidBatchOutput(ValueError):
    pass


class StatefulQueryLeakageError(ValueError):
    def __init__(self, message: str, spot_check: dict | None = None):
        super().__init__(message)
        self.spot_check = spot_check


def validate_submission(root: Path):
    root = root.resolve()
    readme = root / "README.md"
    manifest_file = root / "model_manifest.json"
    for required_name, required_path in (("model_manifest.json", manifest_file), ("README.md", readme)):
        if not required_path.is_file() or required_path.is_symlink():
            raise FileNotFoundError(
                f"submission/{required_name} is required and must not be a symlink"
            )
    for entry_name in ("infer.sh", "serve.sh", "predict_server.py"):
        entry_path = root / entry_name
        if entry_path.is_symlink():
            raise FileNotFoundError(
                f"submission/{entry_name} must not be a symlink"
            )
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError(f"submission symlink escapes package: {path.relative_to(root)}")
    manifest = json.loads(manifest_file.read_text())
    required_fields = {"schema", "checkpoint", "model_config", "training_provenance"}
    allowed_fields = required_fields | {"streaming_entrypoint"}
    if not required_fields.issubset(manifest) or not set(manifest).issubset(allowed_fields):
        raise ValueError("invalid model manifest fields")
    if manifest["schema"] != "massspecgym-retrieval-model/v1":
        raise ValueError("invalid model manifest schema")
    if "streaming_entrypoint" in manifest:
        stream_entry = contained(root, manifest["streaming_entrypoint"])
        if stream_entry.is_symlink():
            raise ValueError("streaming_entrypoint must not be a symlink")
    has_entrypoint = (
        (root / "infer.sh").is_file()
        or (root / "serve.sh").is_file()
        or (root / "predict_server.py").is_file()
        or "streaming_entrypoint" in manifest
    )
    if not has_entrypoint:
        raise FileNotFoundError(
            "submission/infer.sh (or serve.sh / predict_server.py) is required and must not be a symlink"
        )
    checkpoint = manifest["checkpoint"]
    if (
        set(checkpoint) != {"path", "sha256", "format"}
        or checkpoint["format"] not in {"pytorch", "safetensors"}
        or not isinstance(checkpoint["sha256"], str)
        or len(checkpoint["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint["sha256"])
    ):
        raise ValueError("invalid checkpoint declaration")
    checkpoint_path = contained(root, checkpoint["path"])
    if sha256(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    config = contained(root, manifest["model_config"])
    if not isinstance(json.loads(config.read_text()), dict):
        raise ValueError("model config must contain a JSON object")
    provenance = manifest["training_provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"data", "pretrained_assets"}
        or not isinstance(provenance["data"], str)
        or not isinstance(provenance["pretrained_assets"], list)
    ):
        raise ValueError("invalid training provenance")
    if provenance["pretrained_assets"]:
        raise ValueError("this task release supplies no authorized pretrained assets")
    return manifest


def detect_streaming_command(submission: Path, manifest: dict) -> list[str] | None:
    submission = submission.resolve()
    if "streaming_entrypoint" in manifest:
        entry = contained(submission, manifest["streaming_entrypoint"])
        if entry.suffix == ".py":
            return [sys.executable, str(entry)]
        return ["bash", str(entry)]
    if (submission / "serve.sh").is_file():
        return ["bash", str(submission / "serve.sh")]
    if (submission / "predict_server.py").is_file():
        return [sys.executable, str(submission / "predict_server.py")]
    return None


def kill_uid_processes(uid: int):
    # Repeat until no live process remains, so a process forked during a pass cannot survive
    # into the next batch. Zombies hold no state.
    for _ in range(100):
        alive = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                status = dict(line.split(":", 1) for line in
                              (entry / "status").read_text().splitlines() if ":" in line)
                uids = {int(value) for value in status["Uid"].split()}
                if uid in uids and not status["State"].strip().startswith("Z"):
                    alive = True
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError, ValueError):
                pass
        if not alive:
            return
        time.sleep(0.05)
    raise RuntimeError("Cannot isolate the next batch: an inference process is still running")


def _worker_environment(home: Path) -> dict[str, str]:
    py_bin = str(Path(sys.executable).parent)
    default_path = "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
    path_parts = [py_bin] + [p for p in default_path.split(":") if p != py_bin]
    return {
        "PATH": ":".join(path_parts),
        "HOME": str(home),
        "PYTHONUNBUFFERED": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }


def _make_demote(uid: int, gid: int):
    def demote():
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        # Prevent setuid binaries or file capabilities from recovering privilege.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    return demote


def run_submission(submission: Path, input_root: Path, work: Path, timeout: float,
                   home: Path | None = None):
    agent = pwd.getpwnam("agent")
    output = work / "untrusted-output"
    output.mkdir()
    os.chown(output, agent.pw_uid, agent.pw_gid)
    log = work / "inference.log"
    environment = _worker_environment(home or Path(agent.pw_dir))
    demote = _make_demote(agent.pw_uid, agent.pw_gid)

    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(
                ["bash", str(submission / "infer.sh"), str(input_root), str(output)],
                cwd=submission,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=demote,
            )
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    except subprocess.TimeoutExpired:
        if process:
            os.killpg(process.pid, signal.SIGKILL)
        raise
    finally:
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        kill_uid_processes(agent.pw_uid)
    return output


def seal_output(
    untrusted: Path,
    sealed: Path,
    input_manifest: Path,
    batch_record: Path | None = None,
    stream_record: Path | None = None,
):
    if sealed.exists():
        raise FileExistsError(sealed)
    temporary = sealed.with_name(sealed.name + ".tmp")
    temporary.mkdir(parents=True)
    files = {}
    source = untrusted / "rankings.jsonl"
    if source.is_file() and not source.is_symlink():
        if source.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("ranking output exceeds 512 MiB")
        if not source.resolve().is_relative_to(untrusted.resolve()):
            raise ValueError("ranking output escapes output root")
        destination = temporary / "rankings.jsonl"
        shutil.copyfile(source, destination, follow_symlinks=False)
        destination.chmod(0o444)
        files["rankings.jsonl"] = sha256(destination)
    snapshot = {
        "schema": "massspecgym-retrieval-ranking-snapshot/v1",
        "input_manifest_sha256": sha256(input_manifest),
        "files": files,
    }
    if stream_record is not None:
        snapshot["evaluation_protocol"] = STREAM_PROTOCOL
        snapshot["protocol_mode"] = STREAM_PROTOCOL
        snapshot["stream_record_sha256"] = sha256(stream_record)
        stream_meta = json.loads(stream_record.read_text())
        snapshot["statelessness_spot_check"] = stream_meta["statelessness_spot_check"]
    elif batch_record is not None:
        snapshot["evaluation_protocol"] = REPLAY_PROTOCOL
        snapshot["protocol_mode"] = REPLAY_PROTOCOL
        snapshot["batch_record_sha256"] = sha256(batch_record)
        plan = json.loads(batch_record.read_text())
        snapshot["batch_count"] = plan["batches"]
        snapshot["batch_plan_sha256"] = plan["batch_plan_sha256"]
    write_json(temporary / "snapshot.json", snapshot)
    (temporary / "snapshot.json").chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    ready = sealed.parent / "READY.tmp"
    ready.write_text(sha256(sealed / "snapshot.json") + "\n")
    os.replace(ready, sealed.parent / "READY")


def select_spot_check_queries(
    queries: dict,
    pools: dict,
    max_queries: int = SPOT_CHECK_MAX_QUERIES,
    seed: str = STREAM_PRIVATE_SEED,
) -> tuple[list[str], int]:
    """Deterministically sample up to `max_queries` queries for cold-start verification.

    Stratified to prioritize later occurrences in repeated/shared/isomeric candidate pools,
    then shuffled so Pass 2 never replays Pass 1's query order or shared-pool arrival order.
    Returns `(selected_query_ids, repeated_pool_query_count)`.
    """
    if not queries or max_queries <= 0:
        return [], 0
    target_count = min(max_queries, len(queries))
    rng = random.Random(hashlib.sha256(f"{seed}\0spot-select".encode("utf-8")).digest())

    by_pool: dict[str, list[str]] = {}
    pass1_index: dict[str, int] = {}
    for idx, query_id in enumerate(queries):
        pass1_index[query_id] = idx
        pool_id = queries[query_id]["candidate_pool_id"]
        by_pool.setdefault(pool_id, []).append(query_id)

    repeated_pools = {pid for pid, members in by_pool.items() if len(members) >= 2}
    tier1_distinct_repeated_later = []
    tier2_other_repeated_later = []
    tier3_repeated_first = []
    tier4_isomeric_singletons = []
    tier5_other_singletons = []

    for pool_id in sorted(by_pool):
        members = by_pool[pool_id]
        if len(members) >= 2:
            later = list(members[1:])
            rng.shuffle(later)
            # Prioritize the latest/later occurrence from each distinct repeated pool first.
            chosen = members[-1]
            tier1_distinct_repeated_later.append(chosen)
            tier2_other_repeated_later.extend(q for q in later if q != chosen)
            tier3_repeated_first.append(members[0])
        elif len(pools[pool_id]) >= 2:
            tier4_isomeric_singletons.append(members[0])
        else:
            tier5_other_singletons.append(members[0])

    for tier in (
        tier1_distinct_repeated_later,
        tier2_other_repeated_later,
        tier3_repeated_first,
        tier4_isomeric_singletons,
        tier5_other_singletons,
    ):
        rng.shuffle(tier)

    selected = (
        tier1_distinct_repeated_later
        + tier2_other_repeated_later
        + tier3_repeated_first
        + tier4_isomeric_singletons
        + tier5_other_singletons
    )[:target_count]
    rng.shuffle(selected)

    # If multiple queries from the same repeated pool are in `selected`, ensure the latest
    # Pass-1 occurrence is dispatched BEFORE the earliest Pass-1 occurrence in Pass 2 so
    # within-pool arrival order is strictly inverted even when target_count == len(queries).
    positions_by_pool: dict[str, list[int]] = {}
    for pos, qid in enumerate(selected):
        pid = queries[qid]["candidate_pool_id"]
        if pid in repeated_pools:
            positions_by_pool.setdefault(pid, []).append(pos)
    for pid, positions in positions_by_pool.items():
        if len(positions) >= 2:
            ordered_qids = sorted(
                (selected[pos] for pos in positions),
                key=lambda qid: pass1_index[qid],
                reverse=True,
            )
            for pos, qid in zip(positions, ordered_qids):
                selected[pos] = qid

    pass1_prefix = list(queries)[: len(selected)]
    if len(selected) >= 2 and selected == pass1_prefix:
        selected[0], selected[1] = selected[1], selected[0]

    repeated_count = sum(1 for qid in selected if queries[qid]["candidate_pool_id"] in repeated_pools)
    return selected, repeated_count


def _build_stream_request(
    anon_query_id: str,
    orig_query_id: str,
    row: dict,
    pool: dict[str, str],
    seed: str,
    pass_tag: str,
    step_index: int,
) -> dict:
    candidates = [
        {"candidate_id": cid, "smiles": smiles}
        for cid, smiles in sorted(pool.items())
    ]
    rng = random.Random(
        hashlib.sha256(
            f"{seed}\0{pass_tag}\0{step_index}\0{orig_query_id}".encode("utf-8")
        ).digest()
    )
    rng.shuffle(candidates)
    return {
        "query_id": anon_query_id,
        "mzs": row["mzs"],
        "intensities": row["intensities"],
        "precursor_mz": row["precursor_mz"],
        "adduct": row["adduct"],
        "instrument_type": row["instrument_type"],
        "collision_energy": row["collision_energy"],
        "candidates": candidates,
    }


def _read_stream_line(
    stdout_fd: int,
    buffer: bytearray,
    deadline: float,
    command: list[str],
    timeout: float,
    process: subprocess.Popen,
) -> bytes:
    with selectors.DefaultSelector() as selector:
        selector.register(stdout_fd, selectors.EVENT_READ)
        while True:
            newline_pos = buffer.find(b"\n")
            if newline_pos != -1:
                line = bytes(buffer[:newline_pos])
                del buffer[: newline_pos + 1]
                # Reject unsolicited pipelined lines emitted before the next query is sent.
                if buffer.strip():
                    raise InvalidBatchOutput(
                        "streaming worker emitted unsolicited extra output for a single query"
                    )
                return line
            if len(buffer) > MAX_STREAM_LINE_BYTES:
                raise InvalidBatchOutput("streaming response line exceeds 4 MiB")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            events = selector.select(timeout=min(remaining, 1.0))
            if not events:
                continue
            chunk = os.read(stdout_fd, 65536)
            if not chunk:
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise subprocess.TimeoutExpired(command, timeout) from error
                if code:
                    raise subprocess.CalledProcessError(code, command)
                raise InvalidBatchOutput(
                    "streaming worker closed stdout before returning a ranking"
                )
            buffer.extend(chunk)


def _validate_stream_response(raw_line: bytes, expected_query_id: str, expected_pool_ids: set[str]) -> list[str]:
    try:
        response = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidBatchOutput(f"invalid JSON in streaming response: {error}") from error
    if not isinstance(response, dict) or set(response) != {"query_id", "ranking"}:
        raise InvalidBatchOutput(
            "streaming response must contain strictly {'query_id', 'ranking'} with no extra fields"
        )
    if response["query_id"] != expected_query_id:
        raise InvalidBatchOutput(
            f"streaming response query_id mismatch: expected {expected_query_id!r}, got {response['query_id']!r}"
        )
    ranking = response["ranking"]
    if (
        not isinstance(ranking, list)
        or any(not isinstance(cid, str) for cid in ranking)
        or len(ranking) != len(expected_pool_ids)
        or set(ranking) != expected_pool_ids
    ):
        raise InvalidBatchOutput("streaming ranking is not an exact candidate permutation")
    return ranking


def _run_stream_pass(
    query_order: list[str],
    queries: dict,
    pools: dict,
    pass_tag: str,
    seed: str,
    command: list[str],
    deadline: float,
    timeout: float,
    pass_dir: Path,
    spawn_worker,
    cleanup_worker,
) -> dict[str, list[str]]:
    pass_dir.mkdir(mode=0o755)
    log_path = pass_dir / "inference.log"
    process = None
    rankings: dict[str, list[str]] = {}
    try:
        with log_path.open("wb") as log_stream:
            process = spawn_worker(pass_dir, log_stream)
            assert process.stdin is not None and process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            buffer = bytearray()
            for step_index, orig_qid in enumerate(query_order):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                anon_qid = f"sq_{step_index:07d}"
                row = queries[orig_qid]
                pool = pools[row["candidate_pool_id"]]
                payload = _build_stream_request(
                    anon_qid, orig_qid, row, pool, seed, pass_tag, step_index
                )
                encoded = (compact_json(payload) + "\n").encode("utf-8")
                try:
                    process.stdin.write(encoded)
                    process.stdin.flush()
                except BrokenPipeError as error:
                    code = process.poll()
                    if code:
                        raise subprocess.CalledProcessError(code, command) from error
                    raise InvalidBatchOutput("streaming worker closed stdin unexpectedly") from error

                raw_line = _read_stream_line(
                    process.stdout.fileno(),
                    buffer,
                    deadline,
                    command,
                    timeout,
                    process,
                )
                rankings[orig_qid] = _validate_stream_response(
                    raw_line, anon_qid, set(pool)
                )
            try:
                process.stdin.close()
            except OSError:
                pass
            remaining = max(0.1, min(10.0, deadline - time.monotonic()))
            try:
                code = process.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except subprocess.TimeoutExpired:
                # Worker has answered all dispatched queries; forcefully terminate it in finally.
                pass
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                pass
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        if cleanup_worker is not None:
            cleanup_worker()
    return rankings


def execute_streaming(
    input_root: Path,
    queries: dict,
    pools: dict,
    work: Path,
    timeout: float,
    command: list[str],
    spawn_worker,
    cleanup_worker=None,
    private_seed: str = STREAM_PRIVATE_SEED,
    spot_check_max_queries: int = SPOT_CHECK_MAX_QUERIES,
) -> Path:
    """Run persistent 1-at-a-time streaming Pass 1 + cold-start statelessness spot-check Pass 2."""
    logs, merged = work / "logs", work / "merged"
    for path in (logs, merged):
        path.mkdir(mode=0o700)
    manifest_sha = sha256(input_root / "manifest.json")
    effective_seed = hashlib.sha256(
        f"{private_seed}\0{manifest_sha}".encode("utf-8")
    ).hexdigest()
    budget = inference_budget(timeout, 2)
    deadline = time.monotonic() + budget

    pass1_started = time.monotonic()
    pass1_dir = work / "stream-pass1"
    try:
        pass1_rankings = _run_stream_pass(
            list(queries),
            queries,
            pools,
            "pass1",
            effective_seed,
            command,
            deadline,
            timeout,
            pass1_dir,
            spawn_worker,
            cleanup_worker,
        )
    finally:
        if (pass1_dir / "inference.log").exists():
            os.replace(pass1_dir / "inference.log", logs / "stream_pass1.log")
        shutil.rmtree(pass1_dir, ignore_errors=True)
    pass1_seconds = time.monotonic() - pass1_started

    spot_queries, repeated_pool_queries = select_spot_check_queries(
        queries, pools, max_queries=spot_check_max_queries, seed=effective_seed
    )
    pass2_started = time.monotonic()
    pass2_dir = work / "stream-pass2"
    try:
        pass2_rankings = _run_stream_pass(
            spot_queries,
            queries,
            pools,
            "pass2",
            effective_seed,
            command,
            deadline,
            timeout,
            pass2_dir,
            spawn_worker,
            cleanup_worker,
        )
    finally:
        if (pass2_dir / "inference.log").exists():
            os.replace(pass2_dir / "inference.log", logs / "stream_pass2.log")
        shutil.rmtree(pass2_dir, ignore_errors=True)
    pass2_seconds = time.monotonic() - pass2_started

    mismatched = [
        qid for qid in spot_queries if pass2_rankings.get(qid) != pass1_rankings.get(qid)
    ]
    spot_summary = {
        "checked_queries": len(spot_queries),
        "repeated_pool_queries": repeated_pool_queries,
        "mismatched_queries": len(mismatched),
        "passed": len(mismatched) == 0,
    }
    write_json(work / "stream.json", {
        "schema": "massspecgym-retrieval-stream/v2",
        "protocol_mode": STREAM_PROTOCOL,
        "queries": len(queries),
        "inference_budget_seconds": budget,
        "pass1_seconds": pass1_seconds,
        "spot_check_seconds": pass2_seconds,
        "statelessness_spot_check": spot_summary,
    })
    if mismatched:
        raise StatefulQueryLeakageError(
            f"cold-start statelessness spot-check failed on {len(mismatched)}/{len(spot_queries)} "
            f"sampled queries (first mismatch: {mismatched[0]})",
            spot_check=spot_summary,
        )

    with (merged / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for query_id in queries:
            stream.write(
                compact_json({"query_id": query_id, "ranking": pass1_rankings[query_id]}) + "\n"
            )
    return merged


def run_streaming(
    submission: Path,
    command: list[str],
    input_root: Path,
    queries: dict,
    pools: dict,
    work: Path,
    timeout: float,
) -> Path:
    """Run persistent single-spectrum streaming as isolated UID 10001 + cold-start spot-check."""
    agent = pwd.getpwnam("agent")
    for name in INPUT_FILES:
        if readable_by(input_root / name, agent.pw_uid, agent.pw_gid):
            raise RuntimeError(f"the inference UID can read the full test input {input_root / name}")
    writable = isolate_agent_files(agent.pw_uid, agent.pw_gid, skip=(work,))
    demote = _make_demote(agent.pw_uid, agent.pw_gid)

    def spawn_worker(pass_dir: Path, log_stream) -> subprocess.Popen:
        home = pass_dir / "home"
        home.mkdir(mode=0o700)
        os.chown(home, agent.pw_uid, agent.pw_gid)
        return subprocess.Popen(
            command,
            cwd=submission,
            env=_worker_environment(home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_stream,
            start_new_session=True,
            preexec_fn=demote,
        )

    def cleanup_worker():
        kill_uid_processes(agent.pw_uid)
        purge_agent_state(agent.pw_uid, agent.pw_gid, writable)

    return execute_streaming(
        input_root, queries, pools, work, timeout, command, spawn_worker, cleanup_worker
    )


def plan_batches(queries: dict) -> list:
    """Batch j holds the j-th spectrum (fixed seeded order) of every pool with more than j spectra.

    No batch holds two spectra of one candidate pool and every query is in exactly one batch. The
    number of batches is the largest pool size, the fewest with this property. Queries keep their
    input order inside a batch.
    """
    position, by_pool = {}, {}
    for index, (query_id, query) in enumerate(queries.items()):
        position[query_id] = index
        by_pool.setdefault(query["candidate_pool_id"], []).append(query_id)
    batches = []
    for members in by_pool.values():
        members.sort(key=lambda query_id: hashlib.sha256(
            f"{BATCH_ORDER_SEED}\0{query_id}".encode("utf-8")).hexdigest())
        for rank, query_id in enumerate(members):
            if rank == len(batches):
                batches.append([])
            batches[rank].append(query_id)
    return [sorted(batch, key=position.__getitem__) for batch in batches]


def keyed_lines(path: Path, key: str) -> dict:
    # Original rows of an input file that validate_input has already checked.
    lines = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                lines[json.loads(line)[key]] = line.rstrip("\n") + "\n"
    return lines


def write_batch_input(destination: Path, manifest: dict, batch: list, queries: dict,
                      query_lines: dict, pool_lines: dict):
    """A fresh input root holding only this batch's queries and the pools they reference."""
    destination.mkdir(mode=0o755)
    pool_ids = {queries[query_id]["candidate_pool_id"] for query_id in batch}
    with (destination / "queries.jsonl").open("w", encoding="utf-8") as stream:
        stream.writelines(query_lines[query_id] for query_id in batch)
    with (destination / "candidate_pools.jsonl").open("w", encoding="utf-8") as stream:
        stream.writelines(line for pool_id, line in pool_lines.items() if pool_id in pool_ids)
    write_json(destination / "manifest.json", {
        **manifest,
        "counts": {"queries": len(batch), "candidate_pools": len(pool_ids)},
        "files": {name: sha256(destination / name) for name in INPUT_FILES},
    })


def read_batch_rankings(output: Path, batch: list):
    """Rows written by one invocation, by query ID; None when it wrote no ranking file.

    The grader validates and scores the concatenated rows as before. A row for a query outside the
    invocation's own input, a duplicate or an unparsable row invalidates the whole output.
    """
    source = output / "rankings.jsonl"
    if not source.is_file() or source.is_symlink():
        return None
    if source.stat().st_size > 512 * 1024 * 1024:
        raise InvalidBatchOutput("ranking output exceeds 512 MiB")
    if not source.resolve().is_relative_to(output.resolve()):
        raise InvalidBatchOutput("ranking output escapes output root")
    allowed, rows = set(batch), {}
    try:
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = row.get("query_id") if isinstance(row, dict) else None
                if not isinstance(query_id, str) or query_id not in allowed or query_id in rows:
                    raise InvalidBatchOutput("unknown or duplicate query ranking in a batch output")
                rows[query_id] = line.strip()
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidBatchOutput(f"invalid ranking output: {error}") from error
    return rows


def readable_by(path: Path, uid: int, gid: int) -> bool:
    # Mode-bit check for a process with this UID/GID and no supplementary groups.
    def allowed(info, user, group, other):
        if info.st_uid == uid:
            return bool(info.st_mode & user)
        return bool(info.st_mode & (group if info.st_gid == gid else other))
    path = path.resolve()
    return all(allowed(os.stat(parent), stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH)
               for parent in path.parents) and allowed(
        os.stat(path), stat.S_IRUSR, stat.S_IRGRP, stat.S_IROTH)


def isolate_agent_files(uid: int, gid: int, skip=()) -> dict:
    """Prepare the container so that no file written by one invocation reaches the next.

    Entries owned by the inference UID are handed to root with their read access kept, files it
    could rewrite in place lose group/other write permission, and directories other users may
    write become sticky. Returns the directories in which the UID can still create entries, with
    their initial children; purge_agent_state empties them of everything else after each batch.
    """
    skip = {Path("/proc"), Path("/sys"), *map(Path, skip)}
    no_write = ~(stat.S_IWGRP | stat.S_IWOTH)

    def writable(info):
        return (info.st_uid == uid or bool(info.st_mode & stat.S_IWOTH)
                or (info.st_gid == gid and bool(info.st_mode & stat.S_IWGRP)))

    directories = {}
    for directory, subdirectories, files in os.walk("/"):
        here = Path(directory)
        subdirectories[:] = [name for name in subdirectories if here / name not in skip]
        links = [here / name for name in subdirectories if (here / name).is_symlink()]
        for path in [here, *(here / name for name in files), *links]:
            try:
                info = os.lstat(path)
                if info.st_uid == uid:
                    os.lchown(path, 0, 0)
                    if not stat.S_ISLNK(info.st_mode):
                        mode = stat.S_IMODE(info.st_mode) & ~(stat.S_ISUID | stat.S_ISGID)
                        if mode & stat.S_IRUSR:
                            mode |= stat.S_IRGRP | stat.S_IROTH
                        if mode & stat.S_IXUSR:
                            mode |= stat.S_IXGRP | stat.S_IXOTH
                        os.chmod(path, mode & no_write)
                    info = os.lstat(path)
                if stat.S_ISDIR(info.st_mode) and writable(info):
                    os.chmod(path, stat.S_IMODE(info.st_mode) | stat.S_ISVTX)
                    directories[path] = {entry.name for entry in os.scandir(path)}
                elif stat.S_ISREG(info.st_mode) and writable(info):
                    os.chmod(path, stat.S_IMODE(info.st_mode) & no_write)
            except FileNotFoundError:
                pass
            except OSError as error:  # a read-only mount is not writable by the UID either
                if error.errno != errno.EROFS:
                    raise
    return directories


def ipc_objects(uid: int) -> list:
    objects = []
    for kind, column in (("shm", "shmid"), ("msg", "msqid"), ("sem", "semid")):
        try:
            lines = (Path("/proc/sysvipc") / kind).read_text().splitlines()
        except FileNotFoundError:
            continue
        header = lines[0].split() if lines else []
        for line in lines[1:]:
            fields = dict(zip(header, line.split()))
            if uid in (int(fields["uid"]), int(fields["cuid"])):
                objects.append((kind, int(fields[column])))
    return objects


def remove_ipc_objects(uid: int, gid: int):
    # Container root lacks CAP_IPC_OWNER, so a child with the creator's UID removes them.
    objects = ipc_objects(uid)
    if not objects:
        return
    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            libc = ctypes.CDLL(None, use_errno=True)
            for kind, identifier in objects:  # cmd 0 = IPC_RMID
                {"shm": lambda: libc.shmctl(identifier, 0, None),
                 "msg": lambda: libc.msgctl(identifier, 0, None),
                 "sem": lambda: libc.semctl(identifier, 0, 0)}[kind]()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    if ipc_objects(uid):
        raise RuntimeError("System V IPC objects of the inference UID could not be removed")


def purge_agent_state(uid: int, gid: int, directories: dict):
    """Remove every entry created since isolate_agent_files and the UID's System V IPC objects."""
    for directory, initial in directories.items():
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            continue
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if entry.name in initial and info.st_uid != uid:
                continue
            if stat.S_ISDIR(info.st_mode):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
    remove_ipc_objects(uid, gid)


def run_batches(submission: Path, input_root: Path, manifest: dict, queries: dict,
                work: Path, timeout: float) -> Path:
    """Run inference once per batch of plan_batches; return a directory with the merged rankings.

    Every invocation is a fresh infer.sh process with a fresh input root (only its queries and
    their pools), output directory and home directory, and files it leaves anywhere else are
    removed before the next one starts. The timeout covers all invocations. Rows are merged in
    input order; when an invocation writes no ranking file, the merged file is missing, as a
    missing file is for a single invocation.
    """
    agent = pwd.getpwnam("agent")
    for name in INPUT_FILES:
        if readable_by(input_root / name, agent.pw_uid, agent.pw_gid):
            raise RuntimeError(f"the inference UID can read the full test input {input_root / name}")
    writable = isolate_agent_files(agent.pw_uid, agent.pw_gid, skip=(work,))

    def invoke(input_directory, current, remaining):
        home = current / "home"
        home.mkdir(mode=0o700)
        os.chown(home, agent.pw_uid, agent.pw_gid)
        return run_submission(submission, input_directory, current, remaining, home=home)

    def cleanup():
        purge_agent_state(agent.pw_uid, agent.pw_gid, writable)

    return execute_batches(input_root, manifest, queries, work, timeout, invoke, cleanup)


def execute_batches(input_root: Path, manifest: dict, queries: dict, work: Path, timeout: float,
                    invoke, cleanup=None) -> Path:
    """Shared batch inputs, deadlines and ranking merge for Docker and rootless operator replay."""
    batches = plan_batches(queries)
    query_lines = keyed_lines(input_root / "queries.jsonl", "query_id")
    pool_lines = keyed_lines(input_root / "candidate_pools.jsonl", "candidate_pool_id")
    logs, merged = work / "logs", work / "merged"
    for path in (logs, merged):
        path.mkdir(mode=0o700)
    deadline = time.monotonic() + inference_budget(timeout, len(batches))
    rows, record, complete = {}, [], True
    try:
        for index, batch in enumerate(batches):
            current = work / "batch"
            current.mkdir(mode=0o755)
            started = time.monotonic()
            try:
                write_batch_input(current / "input", manifest, batch, queries, query_lines, pool_lines)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("infer.sh", timeout)
                try:
                    output = invoke(current / "input", current, remaining)
                finally:
                    if (current / "inference.log").exists():
                        os.replace(current / "inference.log", logs / f"{index:04d}.log")
                batch_rows = read_batch_rankings(output, batch)
            finally:
                shutil.rmtree(current)
                if cleanup is not None:
                    cleanup()
                record.append({"batch": index, "queries": len(batch),
                               "seconds": time.monotonic() - started})
            if batch_rows is None:
                complete = False
                break
            rows.update(batch_rows)
    finally:
        write_json(work / "batches.json", {
            "schema": "massspecgym-retrieval-batches/v1", "seed": BATCH_ORDER_SEED,
            "rule": "at most one spectrum per candidate pool per invocation",
            "batches": len(batches), "queries": len(queries),
            "batch_plan_sha256": hashlib.sha256(compact_json(batches).encode()).hexdigest(),
            "inference_budget_seconds": inference_budget(timeout, len(batches)), "invocations": record,
        })
    if complete:
        with (merged / "rankings.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(rows[query_id] + "\n" for query_id in queries if query_id in rows)
    return merged


def prepare_trusted_directories(output: Path, sealed: Path):
    for path, mode in ((output, 0o755), (sealed.parent, 0o700)):
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(mode)
    # Harbor also accepts reward.txt, even if test.sh returns nonzero. Neither
    # stale results nor files written by the inference UID may become rewards.
    for name in ("reward.json", "reward.txt", "report.json"):
        (output / name).unlink(missing_ok=True)


def failed_submission(
    output: Path,
    queries: int,
    status: str,
    error: Exception,
    protocol: str = REPLAY_PROTOCOL,
    spot_check: dict | None = None,
):
    metrics = {"hit_rate_at_1": 0.0, "hit_rate_at_5": 0.0,
               "hit_rate_at_20": 0.0, "mces_at_1": 100.0}
    report = {
        "schema": "massspecgym-retrieval-report/v1",
        "evaluation_protocol": protocol,
        "protocol_mode": protocol,
        "status": status,
        "model_quality_measured": False,
        "queries": queries,
        "valid_rankings": 0,
        "invalid_or_missing_rankings": queries,
        "global_error": str(error),
        "metrics": metrics,
        "score": metrics["hit_rate_at_1"],
        "reward": metrics["hit_rate_at_1"],
        "benchmark_score": normalize(metrics["hit_rate_at_1"]),
        "metric_directions": {name: "minimize" if name == "mces_at_1" else "maximize"
                              for name in metrics},
    }
    if spot_check is not None:
        report["statelessness_spot_check"] = spot_check
    write_json(output / "report.json", report)
    write_json(output / "reward.json", {
        "reward": metrics["hit_rate_at_1"],
        "score": metrics["hit_rate_at_1"],
    })


def evaluate(args):
    started = time.monotonic()
    prepare_trusted_directories(args.output, args.sealed)
    manifest, queries, pools = validate_input(args.input)
    try:
        sub_manifest = validate_submission(args.submission)
    except (OSError, ValueError, TypeError, KeyError) as error:
        failed_submission(args.output, len(queries), "invalid_submission", error)
        return
    stream_cmd = detect_streaming_command(args.submission, sub_manifest or {})
    protocol = STREAM_PROTOCOL if stream_cmd is not None else REPLAY_PROTOCOL
    work = args.output / "main"
    work.mkdir()
    try:
        if stream_cmd is not None:
            untrusted = run_streaming(
                args.submission, stream_cmd, args.input, queries, pools, work, args.timeout
            )
        else:
            untrusted = run_batches(
                args.submission, args.input, manifest, queries, work, args.timeout
            )
    except StatefulQueryLeakageError as error:
        failed_submission(
            args.output,
            len(queries),
            "stateful_query_leakage",
            error,
            protocol=protocol,
            spot_check=error.spot_check,
        )
        return
    except subprocess.TimeoutExpired as error:
        failed_submission(args.output, len(queries), "inference_timeout", error, protocol=protocol)
        return
    except subprocess.CalledProcessError as error:
        failed_submission(args.output, len(queries), "inference_failed", error, protocol=protocol)
        return
    except InvalidBatchOutput as error:
        failed_submission(args.output, len(queries), "invalid_output", error, protocol=protocol)
        return
    try:
        if stream_cmd is not None:
            seal_output(
                untrusted, args.sealed, args.input / "manifest.json",
                None, work / "stream.json",
            )
        else:
            seal_output(
                untrusted, args.sealed, args.input / "manifest.json",
                work / "batches.json",
            )
    except ValueError as error:
        failed_submission(args.output, len(queries), "invalid_output", error, protocol=protocol)
        return
    if not args.no_wait:
        results = args.sealed.parent / "results"
        invocations = 2 if stream_cmd is not None else len(plan_batches(queries))
        extra = STARTUP_SECONDS_PER_INVOCATION * invocations
        deadline = min(time.monotonic() + 3600, started + VERIFIER_BASE_SECONDS + extra)
        while not (results / "reward.json").is_file():
            if (results / "error.json").is_file():
                raise RuntimeError(f"trusted grader failed: {(results / 'error.json').read_text()}")
            if time.monotonic() >= deadline:
                raise TimeoutError("trusted grader did not publish reward.json")
            time.sleep(1)
        for name in ("report.json", "reward.json"):
            shutil.copyfile(results / name, args.output / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--input", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--timeout", type=int, default=18000,
                        help="compute part of the inference budget; see inference_budget")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("the trusted verifier must run as root to isolate inference UID 10001")
    if not 0 < args.timeout <= 18000:
        parser.error("inference timeout must be in 1..18000 seconds")
    try:
        evaluate(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("reward.json", "reward.txt"):
            (args.output / name).unlink(missing_ok=True)
        write_json(args.output / "report.json", {
            "schema": "massspecgym-retrieval-report/v1",
            "status": "infrastructure_error",
            "evaluation_protocol": REPLAY_PROTOCOL,
            "protocol_mode": REPLAY_PROTOCOL,
            "model_quality_measured": False,
            "global_error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()

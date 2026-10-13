from __future__ import annotations

import os
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import signal
import secrets
import subprocess
import sys
import time
from typing import Any

from .io import read_json, update_json, write_json_atomic
from .reference import (
    CODEX_CONTROL_PLANE_HOSTS,
    load_reference_policy,
    provider_endpoint_hostname,
    with_harness_gateway_host,
)
from .security import validate_auth_file
from .snapshot import create_snapshot, latest_snapshot
from .task import (
    NetworkMode,
    harness_llm_environment,
    resolve_compute_profile,
    stage_task,
    task_gpu_count,
    task_gpu_types,
    validate_task,
)


ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
EXECUTABLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\Z")
TRIAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
ACTIVE_STATES = {"starting", "running"}
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


def _new_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_gpu_count(task_dir: Path, value: str) -> int:
    if value == "task":
        return task_gpu_count(task_dir)
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("--gpus must be 'task' or a non-negative integer") from error
    if count < 0:
        raise ValueError("--gpus must be non-negative")
    return count


def start_run(
    *,
    task_dir: Path,
    run_dir: Path,
    data_dir: Path | None,
    agent: str,
    model: str | None,
    agent_kwargs: dict[str, Any],
    network: NetworkMode,
    allowed_hosts: list[str],
    gpus: str | None,
    snapshot_interval: float,
    development_image: str | None,
    verifier_image: str | None,
    auth_file: Path | None,
    agent_bins: dict[str, str] | None = None,
    runtime_env: dict[str, str] | None = None,
    data_env_var: str = "FOLDBENCH_DATA_DIR",
    duration_seconds: float | None = None,
    compute_profile: str | None = None,
    cpus: int | None = None,
    memory_mb: int | None = None,
    storage_mb: int | None = None,
) -> dict:
    task_dir = validate_task(task_dir)
    profile = resolve_compute_profile(task_dir, compute_profile)
    resolved_gpus = gpus if gpus is not None else profile.get("gpus", "task")
    resolved_development_image = development_image or profile.get("development_image")
    resolved_verifier_image = verifier_image or profile.get("verifier_image")
    resolved_cpus = cpus if cpus is not None else profile.get("cpus")
    resolved_memory_mb = (
        memory_mb if memory_mb is not None else profile.get("memory_mb")
    )
    resolved_storage_mb = (
        storage_mb if storage_mb is not None else profile.get("storage_mb")
    )
    for name, value in (
        ("cpus", resolved_cpus),
        ("memory-mb", resolved_memory_mb),
        ("storage-mb", resolved_storage_mb),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive")
    if snapshot_interval <= 0:
        raise ValueError("Snapshot interval must be positive")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("Session duration must be positive")
    gpu_count = resolve_gpu_count(task_dir, str(resolved_gpus))
    gpu_types = task_gpu_types(task_dir) if gpu_count else []
    data_path = data_dir.expanduser().resolve() if data_dir else None
    if not ENV_NAME.fullmatch(data_env_var):
        raise ValueError(f"Invalid --data-env-var name: {data_env_var}")
    if data_path is not None and not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    reference_policy = load_reference_policy(task_dir, data_path)
    if reference_policy is not None:
        if network != "allowlist":
            raise ValueError(
                "This task requires --network allowlist so its audited biomedical "
                "source boundary is enforced"
            )
        unexpected_hosts = set(allowed_hosts) - CODEX_CONTROL_PLANE_HOSTS
        if unexpected_hosts:
            raise ValueError(
                "Biomedical task --allow-host values are limited to the Codex "
                f"control plane; rejected: {', '.join(sorted(unexpected_hosts))}"
            )
        allowed_hosts = sorted(set(allowed_hosts) | set(reference_policy["hosts"]))
        if profile.get("harness_llm_provider"):
            allowed_hosts = with_harness_gateway_host(allowed_hosts)
    runtime_values = dict(runtime_env or {})
    if reference_policy is not None and profile.get("harness_llm_provider"):
        # Development intentionally defaults to the live, profile-bound gateway.
        runtime_values.setdefault("BIOMED_GATEWAY_ONLINE", "1")
        runtime_values.setdefault(
            "BIOMED_GATEWAY_ADMIN_TOKEN", secrets.token_urlsafe(32)
        )
    provider_host = provider_endpoint_hostname(
        profile.get("harness_llm_provider"), runtime_values
    )
    if auth_file is not None:
        auth_file = validate_auth_file(auth_file)
    resolved_agent_bins: dict[str, str] = {}
    for name, raw_path in (agent_bins or {}).items():
        if not EXECUTABLE_NAME.fullmatch(name):
            raise ValueError(f"Invalid agent executable name: {name}")
        binary = Path(raw_path).expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"Agent executable does not exist: {binary}")
        if not os.access(binary, os.X_OK):
            raise PermissionError(f"Agent executable is not executable: {binary}")
        resolved_agent_bins[name] = str(binary)

    # All policy and input validation happens before creating the run directory.
    run_dir = _new_directory(run_dir)

    runtime_task = stage_task(
        task_dir,
        run_dir / "runtime-task",
        network=network,
        allowed_hosts=allowed_hosts,
        gpus=gpu_count,
        development_image=resolved_development_image,
        verifier_image=resolved_verifier_image,
    )
    live_submission = run_dir / "live/submission"
    live_submission.mkdir(parents=True)
    starter = runtime_task / "environment/starter"
    if starter.is_dir():
        shutil.copytree(starter, live_submission, dirs_exist_ok=True, symlinks=True)

    config = {
        "schema_version": "agent-runner-run/v1",
        "task_dir": str(task_dir),
        "runtime_task_dir": str(runtime_task),
        "run_dir": str(run_dir),
        "data_dir": str(data_path) if data_path else None,
        "data_env_var": data_env_var,
        "agent": agent,
        "model": model,
        "agent_kwargs": agent_kwargs,
        "network": network,
        "allowed_hosts": allowed_hosts,
        "gpus": gpu_count,
        "gpu_types": gpu_types,
        "cpus": resolved_cpus,
        "memory_mb": resolved_memory_mb,
        "storage_mb": resolved_storage_mb,
        "compute_profile": profile["name"],
        "harness_llm_provider": profile.get("harness_llm_provider"),
        "harness_llm_model": profile.get("harness_llm_model"),
        "harness_llm_max_model_calls_per_case": profile.get(
            "harness_llm_max_model_calls_per_case"
        ),
        "harness_llm_development_total_tokens": profile.get(
            "harness_llm_development_total_tokens"
        ),
        "harness_llm_max_output_tokens_per_call": profile.get(
            "harness_llm_max_output_tokens_per_call"
        ),
        "biomedical_source_allowlist_revision": (
            reference_policy["revision"] if reference_policy else None
        ),
        "biomedical_source_allowlist_sha256": (
            reference_policy["sha256"] if reference_policy else None
        ),
        "biomedical_source_hosts": (
            reference_policy["hosts"] if reference_policy else []
        ),
        "biomedical_reference_dir": (
            reference_policy["reference_dir"] if reference_policy else None
        ),
        "provider_endpoint_hostname": provider_host,
        "snapshot_interval": snapshot_interval,
        "duration_seconds": duration_seconds,
        "development_image": resolved_development_image,
        "verifier_image": resolved_verifier_image,
        "agent_bins": resolved_agent_bins,
        "runtime_env_names": sorted((runtime_env or {}).keys()),
    }
    # Authentication paths and environment values are intentionally not persisted.
    write_json_atomic(run_dir / "run.json", config)
    write_json_atomic(
        run_dir / "state.json",
        {"schema_version": "agent-runner-state/v1", "status": "starting"},
    )

    child_env = os.environ.copy()
    child_env.update(runtime_values)
    child_env["BIOMED_PROVIDER_ENDPOINT_HOST"] = provider_host or "blocked.invalid"
    if profile.get("harness_llm_provider"):
        child_env.update(
            harness_llm_environment(
                profile, profile["harness_llm_development_total_tokens"]
            )
        )
    if data_path is not None:
        child_env[data_env_var] = str(data_path)
    if auth_file is not None:
        child_env["CODEX_AUTH_JSON_PATH"] = str(auth_file)

    log_path = run_dir / "runner.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "agent_runner.worker", str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )
    write_json_atomic(run_dir / "pid.json", _process_record(process.pid))
    return {**config, "pid": process.pid, "state_path": str(run_dir / "state.json")}


def _process_details(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        fields = raw.rsplit(")", 1)[1].strip().split()
        return fields[0], int(fields[19])
    except (IndexError, ValueError):
        return None


def _process_record(pid: int) -> dict[str, int]:
    details = _process_details(pid)
    return {"pid": pid, "start_ticks": details[1] if details else -1}


def process_alive(pid: int, start_ticks: int | None = None) -> bool:
    # kill(pid, 0) also succeeds for an exited child that has not been reaped.
    # Treat Linux zombies as stopped so detached-session status and stop do not
    # wait for their timeout merely because the short-lived launcher still lives.
    details = _process_details(pid)
    if details is not None:
        if details[0] == "Z":
            return False
        if start_ticks is not None and start_ticks >= 0 and details[1] != start_ticks:
            return False
    elif start_ticks is not None and start_ticks >= 0:
        # A recorded Linux identity must never be accepted without matching
        # /proc. On platforms without /proc, records use -1 and fall through to
        # the portable signal probe below.
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def run_status(run_dir: Path) -> dict:
    run_dir = run_dir.expanduser().resolve()
    state = read_json(run_dir / "state.json")
    pid_record = (
        read_json(run_dir / "pid.json") if (run_dir / "pid.json").is_file() else {}
    )
    pid = pid_record.get("pid")
    state["pid"] = pid
    state["process_alive"] = bool(
        pid and process_alive(int(pid), pid_record.get("start_ticks"))
    )
    latest = latest_snapshot(run_dir / "snapshots")
    state["latest_snapshot"] = latest
    state["live_submission"] = str(run_dir / "live/submission")
    return state


def list_runs(runs_dir: Path, *, active_only: bool = False) -> list[dict]:
    """List runner sessions below one root with a process-aware status."""
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    sessions = []
    for config_path in runs_dir.rglob("run.json"):
        run_dir = config_path.parent
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            continue
        try:
            config = read_json(config_path)
            state = run_status(run_dir)
            recorded_status = state.get("status", "unknown")
            alive = bool(state["process_alive"])
            if alive and recorded_status in {"completed", "failed", "stopped"}:
                effective_status = "orphaned"
            elif not alive and recorded_status in {"starting", "running"}:
                effective_status = "stale"
            else:
                effective_status = recorded_status
            if active_only and not alive:
                continue
            latest = state.get("latest_snapshot") or {}
            sessions.append(
                {
                    "run_dir": str(run_dir),
                    "status": effective_status,
                    "recorded_status": recorded_status,
                    "process_alive": alive,
                    "pid": state.get("pid"),
                    "agent": config.get("agent"),
                    "model": config.get("model"),
                    "task_dir": config.get("task_dir"),
                    "compute_profile": config.get("compute_profile"),
                    "network": config.get("network"),
                    "started_at": state.get("started_at"),
                    "finished_at": state.get("finished_at"),
                    "deadline_at": state.get("deadline_at"),
                    "latest_snapshot_id": latest.get("snapshot_id"),
                    "runner_log": str(run_dir / "runner.log"),
                    "updated_at": datetime.fromtimestamp(
                        state_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
            )
        except (OSError, ValueError) as error:
            if not active_only:
                sessions.append(
                    {
                        "run_dir": str(run_dir),
                        "status": "unreadable",
                        "process_alive": False,
                        "error": str(error),
                    }
                )
    sessions.sort(
        key=lambda item: item.get("started_at") or item.get("updated_at") or "",
        reverse=True,
    )
    return sessions


def snapshot_now(run_dir: Path) -> dict | None:
    run_dir = run_dir.expanduser().resolve()
    return create_snapshot(run_dir / "live/submission", run_dir / "snapshots")


def _docker_output(arguments: list[str]) -> list[str]:
    """Run one non-shelling Docker query/action and return non-empty output lines."""
    try:
        result = subprocess.run(
            ["docker", *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Docker CLI is required to clean up a stale run") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Docker {' '.join(arguments[:2])} failed: {detail or result.returncode}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _trial_names(run_dir: Path, state: dict[str, Any]) -> set[str]:
    """Return only validated Harbor trial names recorded below this run."""
    names: set[str] = set()
    recorded = state.get("trial_name")
    if isinstance(recorded, str) and TRIAL_NAME.fullmatch(recorded):
        names.add(recorded)
    harbor_dir = run_dir / "harbor"
    if harbor_dir.is_dir():
        for path in harbor_dir.iterdir():
            if path.is_dir() and TRIAL_NAME.fullmatch(path.name):
                names.add(path.name)
    return names


def _compose_projects_for_trials(trial_names: set[str]) -> list[str]:
    """Discover Compose projects whose exact trial-name prefix belongs to this run."""
    if not trial_names:
        return []
    projects: set[str] = set()
    format_value = '{{.Label "' + COMPOSE_PROJECT_LABEL + '"}}'
    for list_command in (
        ["container", "ls", "--all"],
        ["network", "ls"],
        ["volume", "ls"],
    ):
        values = _docker_output(
            [*list_command, "--filter", f"label={COMPOSE_PROJECT_LABEL}",
             "--format", format_value]
        )
        for project in values:
            if any(
                project == trial_name or project.startswith(trial_name + "__")
                for trial_name in trial_names
            ):
                projects.add(project)
    return sorted(projects)


def _remove_compose_project(project: str, *, timeout: float) -> None:
    """Remove containers and residual networks/volumes for one exact Compose label."""
    label = f"label={COMPOSE_PROJECT_LABEL}={project}"
    containers = _docker_output(
        ["container", "ls", "--all", "--quiet", "--filter", label]
    )
    if containers:
        _docker_output(
            ["container", "stop", "--time", str(max(1, int(timeout))), *containers]
        )
        _docker_output(["container", "rm", "--force", *containers])
    networks = _docker_output(
        ["network", "ls", "--quiet", "--filter", label]
    )
    if networks:
        _docker_output(["network", "rm", *networks])
    volumes = _docker_output(
        ["volume", "ls", "--quiet", "--filter", label]
    )
    if volumes:
        _docker_output(["volume", "rm", *volumes])


def _append_run_event(run_dir: Path, event: str, **fields: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "info",
        "event": event,
        **fields,
    }
    with (run_dir / "runner.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _stop_stale_run(run_dir: Path, state: dict[str, Any], *, timeout: float) -> dict:
    """Clean Harbor resources after the supervising worker has disappeared."""
    trial_names = _trial_names(run_dir, state)
    projects = _compose_projects_for_trials(trial_names)
    _append_run_event(
        run_dir,
        "stale_cleanup_started",
        trial_names=sorted(trial_names),
        compose_projects=projects,
    )
    for project in projects:
        _remove_compose_project(project, timeout=timeout)
        _append_run_event(run_dir, "stale_compose_project_removed", project=project)
    snapshot = create_snapshot(run_dir / "live/submission", run_dir / "snapshots")
    update_json(
        run_dir / "state.json",
        status="stopped",
        finished_at=datetime.now(timezone.utc).isoformat(),
        stop_reason="stale_worker_cleanup",
        stale_cleanup_projects=projects,
    )
    _append_run_event(
        run_dir,
        "stale_cleanup_finished",
        compose_projects=projects,
        snapshot_id=(snapshot or {}).get("snapshot_id"),
    )
    return run_status(run_dir)


def stop_run(run_dir: Path, *, timeout: float = 45.0, force: bool = False) -> dict:
    run_dir = run_dir.expanduser().resolve()
    pid_path = run_dir / "pid.json"
    pid_record = read_json(pid_path) if pid_path.is_file() else {}
    pid = int(pid_record["pid"]) if pid_record.get("pid") is not None else None
    start_ticks = pid_record.get("start_ticks")
    if pid is None or not process_alive(pid, start_ticks):
        state = read_json(run_dir / "state.json")
        if state.get("status") in ACTIVE_STATES:
            return _stop_stale_run(run_dir, state, timeout=timeout)
        return run_status(run_dir)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while process_alive(pid, start_ticks) and time.monotonic() < deadline:
        time.sleep(0.2)
    if process_alive(pid, start_ticks):
        if not force:
            raise TimeoutError(
                f"Runner {pid} did not stop within {timeout}s; retry with --force"
            )
        os.killpg(pid, signal.SIGKILL)
    state = read_json(run_dir / "state.json")
    if state.get("status") in ACTIVE_STATES:
        return _stop_stale_run(run_dir, state, timeout=timeout)
    return run_status(run_dir)


def materialize(task_dir: Path, data_dir: Path, *, runtime_env: dict[str, str]) -> None:
    task_dir = validate_task(task_dir)
    script = task_dir / "environment/data/materialize.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Task has no materializer: {script}")
    environment = os.environ.copy()
    environment.update(runtime_env)
    result = subprocess.run(
        ["bash", str(script), str(data_dir.expanduser().resolve())], env=environment
    )
    if result.returncode:
        raise RuntimeError(
            f"Task data materializer exited with status {result.returncode}"
        )

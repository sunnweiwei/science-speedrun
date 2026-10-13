from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Iterator

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from .io import read_json, write_json_atomic
from .naming import unique_trial_name
from .reference import (
    load_reference_policy,
    provider_endpoint_hostname,
    with_harness_gateway_host,
)
from .snapshot import verify_snapshot
from .task import (
    harness_llm_environment,
    require_separate_verifier,
    resolve_compute_profile,
    stage_task,
    task_gpu_count,
    validate_resolved_harness_llm_profile,
)


def _evaluation_event(event: str, **fields) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{key: value for key, value in fields.items() if value is not None},
    }
    print(json.dumps(record, sort_keys=True, default=str), file=sys.stderr, flush=True)


async def _progress_loop(trial_dir: Path, stop: asyncio.Event) -> None:
    started = asyncio.get_running_loop().time()
    while not stop.is_set():
        verifier_dir = trial_dir / "verifier"
        verifier_files = []
        if verifier_dir.is_dir():
            try:
                verifier_files = [path for path in verifier_dir.rglob("*") if path.is_file()]
            except OSError:
                verifier_files = []
        try:
            verifier_bytes = sum(path.stat().st_size for path in verifier_files)
        except OSError:
            verifier_bytes = 0
        trial_log = trial_dir / "trial.log"
        try:
            trial_log_bytes = trial_log.stat().st_size
        except OSError:
            trial_log_bytes = 0
        _evaluation_event(
            "evaluation_progress",
            elapsed_seconds=round(asyncio.get_running_loop().time() - started, 1),
            verifier_files=len(verifier_files),
            verifier_bytes=verifier_bytes,
            trial_log_bytes=trial_log_bytes,
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_snapshot(run_dir: Path, snapshot: str) -> Path:
    snapshots = (run_dir / "snapshots").resolve()
    if snapshot == "latest":
        pointer = read_json(snapshots / "latest.json")
        candidate = Path(pointer["path"]).resolve()
    else:
        candidate = (snapshots / snapshot).resolve()
    if not candidate.is_relative_to(snapshots) or not candidate.is_dir():
        raise ValueError(f"Unknown snapshot: {snapshot}")
    verify_snapshot(candidate)
    return candidate


async def _evaluate(
    run_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    data_dir: Path | None,
    gpus: int,
    profile: dict,
    verifier_image: str | None,
    runtime_env: dict[str, str],
) -> dict:
    run = read_json(run_dir / "run.json")
    _evaluation_event("runtime_task_staging", output_dir=str(output_dir))
    has_harness_llm = bool(profile.get("harness_llm_provider"))
    reference_policy = load_reference_policy(Path(run["task_dir"]), data_dir)
    biomedical_hosts = reference_policy["hosts"] if reference_policy else []
    verifier_hosts = (
        with_harness_gateway_host(biomedical_hosts)
        if has_harness_llm else biomedical_hosts
    )
    provider_host = provider_endpoint_hostname(
        profile.get("harness_llm_provider"), runtime_env
    )
    runtime_task = stage_task(
        Path(run["task_dir"]),
        output_dir / "runtime-task",
        network="no-network",
        allowed_hosts=[],
        gpus=gpus,
        development_image=profile.get("development_image") or run.get("development_image"),
        verifier_image=(verifier_image or profile.get("verifier_image") or
                        run.get("verifier_image")),
        verifier_network="allowlist" if verifier_hosts else "no-network",
        verifier_allowed_hosts=verifier_hosts,
    )
    require_separate_verifier(runtime_task)
    _evaluation_event("runtime_task_staged", runtime_task_dir=str(runtime_task))
    snapshot_id = snapshot_dir.name
    trial_name = unique_trial_name("evaluate", output_dir)
    config = TrialConfig(
        task=TaskConfig(path=runtime_task),
        trial_name=trial_name,
        trials_dir=output_dir / "harbor",
        agent=AgentConfig(name="agent_runner.agents:ArtifactReplayAgent"),
        environment=EnvironmentConfig(
            import_path=("agent_runner.environment:AllowlistDockerEnvironment"
                         if verifier_hosts else
                         "agent_runner.environment:NoNetworkDockerEnvironment"),
            override_cpus=profile.get("cpus", run.get("cpus")),
            override_memory_mb=profile.get("memory_mb", run.get("memory_mb")),
            override_storage_mb=profile.get("storage_mb", run.get("storage_mb")),
            override_gpus=0,
            mounts=[
                {
                    "type": "bind",
                    "source": str((snapshot_dir / "submission").resolve()),
                    "target": "/runner-input/submission",
                    "read_only": True,
                    "bind": {"create_host_path": False},
                }
            ] + ([{
                "type": "bind",
                "source": reference_policy["reference_dir"],
                "target": "/reference",
                "read_only": True,
                "bind": {"create_host_path": False},
            }] if reference_policy else []),
            env=({
                "BIOMED_REFERENCE_DATA_DIR": "/reference",
                "BIOMED_SOURCE_ALLOWLIST_PATH": "/reference/source-allowlist.json",
            } if reference_policy else {}),
            delete=True,
        ),
    )
    data_env_var = run.get("data_env_var", "FOLDBENCH_DATA_DIR")
    environment = dict(runtime_env)
    environment.setdefault("BIOMED_GATEWAY_ONLINE", "0")
    if has_harness_llm:
        environment.setdefault(
            "BIOMED_GATEWAY_ADMIN_TOKEN", secrets.token_urlsafe(32)
        )
    environment["BIOMED_PROVIDER_ENDPOINT_HOST"] = provider_host or "blocked.invalid"
    if has_harness_llm:
        environment.update(harness_llm_environment(profile, None))
    if data_dir is not None:
        environment[data_env_var] = str(data_dir)
    trial_dir = (output_dir / "harbor" / trial_name).resolve()
    with _temporary_environment(environment):
        _evaluation_event(
            "trial_creating", trial_name=trial_name, trial_dir=str(trial_dir)
        )
        trial = await Trial.create(config)
        _evaluation_event("trial_created", trial_name=trial_name)
        progress_stop = asyncio.Event()
        progress_task = asyncio.create_task(_progress_loop(trial_dir, progress_stop))
        try:
            result = await trial.run()
        finally:
            progress_stop.set()
            await progress_task
    _evaluation_event(
        "trial_finished",
        trial_name=trial_name,
        status="failed" if result.exception_info else "completed",
    )
    verify_snapshot(snapshot_dir)
    _evaluation_event("snapshot_reverified", snapshot_id=snapshot_id)
    record = {
        "schema_version": "agent-runner-evaluation/v1",
        "snapshot_id": snapshot_id,
        "snapshot_tree_sha256": read_json(snapshot_dir / "manifest.json")[
            "tree_sha256"
        ],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "compute_profile": profile["name"],
        "harness_llm_provider": profile.get("harness_llm_provider"),
        "harness_llm_model": profile.get("harness_llm_model"),
        "harness_llm_limits": ({
            "model_calls_per_case": profile["harness_llm_max_model_calls_per_case"],
            "output_tokens_per_call": profile[
                "harness_llm_max_output_tokens_per_call"
            ],
            "total_tokens_limit": None,
        } if has_harness_llm else None),
        "biomedical_source_allowlist_revision": (
            reference_policy["revision"] if reference_policy else None
        ),
        "biomedical_source_allowlist_sha256": (
            reference_policy["sha256"] if reference_policy else None
        ),
        "biomedical_source_hosts": biomedical_hosts,
        "provider_endpoint_hostname": provider_host,
        "trial_dir": str(trial_dir),
        "rewards": result.verifier_result.rewards if result.verifier_result else None,
        "status": "failed" if result.exception_info else "completed",
        "error": result.exception_info.exception_message
        if result.exception_info
        else None,
        "exception_type": result.exception_info.exception_type
        if result.exception_info
        else None,
    }
    write_json_atomic(output_dir / "evaluation.json", record)
    _evaluation_event(
        "evaluation_record_written",
        status=record["status"],
        evaluation_path=str(output_dir / "evaluation.json"),
    )
    if result.exception_info:
        raise RuntimeError(
            f"External evaluation failed: {result.exception_info.exception_type}: "
            f"{result.exception_info.exception_message}"
        )
    return record


def evaluate_snapshot(
    run_dir: Path,
    *,
    snapshot: str = "latest",
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    gpus: int | None = None,
    verifier_image: str | None = None,
    compute_profile: str | None = None,
    runtime_env: dict[str, str] | None = None,
) -> dict:
    run_dir = run_dir.expanduser().resolve()
    _evaluation_event(
        "evaluation_started",
        run_dir=str(run_dir),
        requested_snapshot=snapshot,
    )
    run = read_json(run_dir / "run.json")
    if compute_profile is None:
        profile = {"name": run.get("compute_profile")}
        for field in (
            "gpus", "cpus", "memory_mb", "storage_mb", "development_image",
            "verifier_image", "harness_llm_provider", "harness_llm_model",
            "harness_llm_max_model_calls_per_case",
            "harness_llm_development_total_tokens",
            "harness_llm_max_output_tokens_per_call",
        ):
            if run.get(field) is not None:
                profile[field] = run[field]
    else:
        profile = resolve_compute_profile(Path(run["task_dir"]), compute_profile)
    validate_resolved_harness_llm_profile(profile)
    profile_gpus = profile.get("gpus", run["gpus"])
    if profile_gpus == "task":
        profile_gpus = task_gpu_count(Path(run["task_dir"]))
    resolved_gpus = int(profile_gpus if gpus is None else gpus)
    snapshot_dir = resolve_snapshot(run_dir, snapshot)
    _evaluation_event(
        "snapshot_verified",
        snapshot_id=snapshot_dir.name,
        snapshot_dir=str(snapshot_dir),
    )
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir
        else run_dir / "evaluations" / snapshot_dir.name
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_data = data_dir.expanduser().resolve() if data_dir else None
    if resolved_data is None and run.get("data_dir"):
        resolved_data = Path(run["data_dir"]).resolve()
    if resolved_data is not None and not resolved_data.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {resolved_data}")
    try:
        return asyncio.run(
            _evaluate(
                run_dir,
                snapshot_dir,
                output_dir,
                resolved_data,
                resolved_gpus,
                profile,
                verifier_image,
                runtime_env or {},
            )
        )
    except BaseException as error:
        _evaluation_event(
            "evaluation_failed",
            exception_type=type(error).__name__,
            error=str(error),
        )
        raise

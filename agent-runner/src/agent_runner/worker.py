from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from .io import read_json, update_json, write_json_atomic
from .naming import unique_trial_name
from .snapshot import create_snapshot
from .task import harness_llm_environment, validate_resolved_harness_llm_profile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    """Emit one immediately flushed, machine-readable lifecycle event."""
    record = {
        "timestamp": _now(),
        "level": level,
        "event": event,
        **{key: value for key, value in fields.items() if value is not None},
    }
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


COMPUTE_BUDGET_INSTRUCTION_MARKER = "<!-- agent-runner-compute-budget/v1 -->"
COMPUTE_BUDGET_INSTRUCTION = f"""

{COMPUTE_BUDGET_INSTRUCTION_MARKER}
## Runtime compute budget (final instruction)

Before planning, training, or evaluating models, read
`/workspace/campaign-budget.json`. It is the authoritative description of this
run's wall-clock deadline, effective accelerator/resource limits, and harness
LLM development budget. A `null` CPU, memory, or storage limit means the runner did not set an explicit limit;
inspect the runtime environment before choosing resource-intensive settings.

Use the available compute budget deliberately. Producing a valid first result
is a milestone, not the stopping condition: measure per-task validation results,
identify weak tasks, and continue improving and rerunning promising approaches
while enough time remains. Preserve sufficient time to write and smoke-test the
final submission. Do not exceed the budget or rely on resources absent from the
JSON file.
"""


def _prepare_compute_budget(
    run_dir: Path,
    config: dict[str, Any],
    deadline: datetime | None,
) -> Path:
    """Materialize and advertise the effective budget for the development agent."""
    budget_path = run_dir / "campaign-budget.json"
    write_json_atomic(
        budget_path,
        {
            "schema_version": "agent-runner-compute-budget/v1",
            "compute_profile": config.get("compute_profile"),
            "wall_time_seconds": config.get("duration_seconds"),
            "deadline_utc": deadline.isoformat() if deadline is not None else None,
            "resources": {
                "gpu_count": int(config.get("gpus") or 0),
                "gpu_types": list(config.get("gpu_types") or []),
                "cpu_count": config.get("cpus"),
                "memory_mb": config.get("memory_mb"),
                "storage_mb": config.get("storage_mb"),
            },
            "harness_llm": ({
                "provider": config.get("harness_llm_provider"),
                "model": config.get("harness_llm_model"),
                "limits": {
                    "model_calls_per_case": config.get(
                        "harness_llm_max_model_calls_per_case"
                    ),
                    "development_total_tokens": config.get(
                        "harness_llm_development_total_tokens"
                    ),
                    "output_tokens_per_call": config.get(
                        "harness_llm_max_output_tokens_per_call"
                    ),
                },
            } if config.get("harness_llm_provider") else None),
            "unspecified_limits_use_runtime_defaults": True,
        },
    )

    instruction_path = Path(config["runtime_task_dir"]) / "instruction.md"
    instruction = instruction_path.read_text()
    if COMPUTE_BUDGET_INSTRUCTION_MARKER not in instruction:
        instruction_path.write_text(
            instruction.rstrip() + COMPUTE_BUDGET_INSTRUCTION + "\n"
        )
    return budget_path


async def _snapshot_loop(run_dir: Path, interval: float, stop: asyncio.Event) -> None:
    last_snapshot_id: str | None = None
    while not stop.is_set():
        try:
            snapshot = create_snapshot(
                run_dir / "live/submission",
                run_dir / "snapshots",
                settle_seconds=min(0.2, interval / 4),
            )
            if snapshot and snapshot.get("snapshot_id") != last_snapshot_id:
                last_snapshot_id = snapshot.get("snapshot_id")
                _log_event(
                    "snapshot_published",
                    snapshot_id=last_snapshot_id,
                    tree_sha256=snapshot.get("tree_sha256"),
                )
        except Exception as error:
            update_json(run_dir / "state.json", snapshot_error=str(error))
            _log_event(
                "snapshot_failed",
                level="error",
                exception_type=type(error).__name__,
                error=str(error),
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def run_worker(run_dir: Path) -> int:
    _log_event("worker_started", pid=os.getpid(), run_dir=str(run_dir))
    try:
        config = read_json(run_dir / "run.json")
    except BaseException as error:
        _log_event(
            "configuration_load_failed",
            level="error",
            exception_type=type(error).__name__,
            error=str(error),
        )
        raise
    validate_resolved_harness_llm_profile(config)
    _log_event(
        "configuration_loaded",
        agent=config.get("agent"),
        model=config.get("model"),
        network=config.get("network"),
        allowed_hosts=config.get("allowed_hosts") or [],
        compute_profile=config.get("compute_profile"),
        gpus=config.get("gpus"),
        gpu_types=config.get("gpu_types") or [],
        cpus=config.get("cpus"),
        memory_mb=config.get("memory_mb"),
        storage_mb=config.get("storage_mb"),
        duration_seconds=config.get("duration_seconds"),
        snapshot_interval=config.get("snapshot_interval"),
        runtime_env_names=config.get("runtime_env_names") or [],
        agent_bin_names=sorted((config.get("agent_bins") or {}).keys()),
        runtime_task_dir=config.get("runtime_task_dir"),
    )
    if config.get("harness_llm_provider"):
        # Reconstruct non-secret gateway configuration from persisted run metadata.
        # This also prevents ambient values from overriding the selected profile.
        profile = {**config, "name": config["compute_profile"]}
        os.environ.update(harness_llm_environment(
            profile, config["harness_llm_development_total_tokens"]
        ))
    stop = asyncio.Event()
    stop_reason: dict[str, str | None] = {"value": None}
    loop = asyncio.get_running_loop()

    def request_stop(reason: str) -> None:
        if not stop.is_set():
            stop_reason["value"] = reason
            _log_event("stop_requested", reason=reason)
            stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            signum,
            request_stop,
            f"signal:{signal.Signals(signum).name}",
        )

    duration_seconds = config.get("duration_seconds")
    deadline_handle = None
    deadline = None
    if duration_seconds is not None:
        duration_seconds = float(duration_seconds)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        update_json(run_dir / "state.json", deadline_at=deadline.isoformat())
        _log_event(
            "deadline_scheduled",
            deadline_at=deadline.isoformat(),
            duration_seconds=duration_seconds,
        )
        deadline_handle = loop.call_later(
            duration_seconds, request_stop, "duration_elapsed"
        )

    budget_path = _prepare_compute_budget(run_dir, config, deadline)
    _log_event(
        "compute_budget_prepared",
        budget_path=str(budget_path),
        deadline_at=deadline.isoformat() if deadline is not None else None,
    )

    mounts = [
        {
            "type": "bind",
            "source": str((run_dir / "live/submission").resolve()),
            "target": "/workspace/submission",
            "bind": {"create_host_path": False},
        },
        {
            "type": "bind",
            "source": str(budget_path.resolve()),
            "target": "/workspace/campaign-budget.json",
            "read_only": True,
            "bind": {"create_host_path": False},
        },
    ]
    for name, source in (config.get("agent_bins") or {}).items():
        mounts.append(
            {
                "type": "bind",
                "source": source,
                "target": f"/usr/local/bin/{name}",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        )
    container_env: dict[str, str] = {}
    if config.get("biomedical_reference_dir"):
        mounts.append(
            {
                "type": "bind",
                "source": config["biomedical_reference_dir"],
                "target": "/reference",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        )
        container_env.update(
            {
                "BIOMED_REFERENCE_DATA_DIR": "/reference",
                "BIOMED_SOURCE_ALLOWLIST_PATH": "/reference/source-allowlist.json",
            }
        )

    environment_import = None
    if config["network"] == "no-network":
        environment_import = "agent_runner.environment:NoNetworkDockerEnvironment"
    elif config["network"] == "allowlist":
        environment_import = "agent_runner.environment:AllowlistDockerEnvironment"

    trial_name = unique_trial_name("development", run_dir)
    update_json(run_dir / "state.json", trial_name=trial_name)
    trial_config = TrialConfig(
        task=TaskConfig(path=Path(config["runtime_task_dir"])),
        trial_name=trial_name,
        trials_dir=run_dir / "harbor",
        agent=AgentConfig(
            name=config["agent"],
            model_name=config.get("model"),
            kwargs=config.get("agent_kwargs") or {},
        ),
        environment=EnvironmentConfig(
            import_path=environment_import,
            override_cpus=config.get("cpus"),
            override_memory_mb=config.get("memory_mb"),
            override_storage_mb=config.get("storage_mb"),
            override_gpus=0,
            mounts=mounts,
            env=container_env,
            delete=True,
        ),
        verifier=VerifierConfig(disable=True),
    )
    _log_event("trial_creating", trial_name=trial_name)
    try:
        trial = await Trial.create(trial_config)
    except BaseException as error:
        _log_event(
            "trial_create_failed",
            level="error",
            trial_name=trial_name,
            exception_type=type(error).__name__,
            error=str(error),
        )
        update_json(
            run_dir / "state.json",
            status="failed",
            finished_at=_now(),
            error=str(error),
            exception_type=type(error).__name__,
        )
        _log_event("worker_finished", status="failed", exit_code=1)
        return 1
    _log_event(
        "trial_created",
        trial_name=trial_name,
        trial_dir=str(run_dir / "harbor" / trial_name),
    )

    async def on_agent_start(_event) -> None:
        update_json(
            run_dir / "state.json",
            status="running",
            started_at=_now(),
            pid=os.getpid(),
        )
        _log_event("agent_started", trial_name=trial_name, pid=os.getpid())

    trial.add_hook(TrialEvent.AGENT_START, on_agent_start)
    snapshot_task = asyncio.create_task(
        _snapshot_loop(run_dir, float(config["snapshot_interval"]), stop)
    )
    trial_task = asyncio.create_task(trial.run())
    stop_task = asyncio.create_task(stop.wait())
    status = "failed"
    exit_code = 1
    try:
        done, _ = await asyncio.wait(
            {trial_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done and stop.is_set() and not trial_task.done():
            _log_event(
                "trial_cancelling",
                trial_name=trial_name,
                reason=stop_reason["value"],
            )
            trial_task.cancel()
            try:
                await trial_task
            except asyncio.CancelledError:
                pass
            status = "stopped"
            exit_code = 0
            _log_event(
                "trial_stopped",
                trial_name=trial_name,
                reason=stop_reason["value"],
            )
        else:
            result = await trial_task
            if result.exception_info is None:
                status = "completed"
                exit_code = 0
                _log_event("trial_completed", trial_name=trial_name)
            else:
                status = "failed"
                _log_event(
                    "trial_failed",
                    level="error",
                    trial_name=trial_name,
                    exception_type=result.exception_info.exception_type,
                    error=result.exception_info.exception_message,
                )
                update_json(
                    run_dir / "state.json",
                    error=result.exception_info.exception_message,
                    exception_type=result.exception_info.exception_type,
                )
    except BaseException as error:
        if (
            isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            or stop.is_set()
        ):
            status = "stopped"
            exit_code = 0
            _log_event(
                "worker_interrupted",
                level="warning",
                trial_name=trial_name,
                reason=stop_reason["value"] or type(error).__name__,
            )
        else:
            _log_event(
                "worker_failed",
                level="error",
                trial_name=trial_name,
                exception_type=type(error).__name__,
                error=str(error),
            )
            update_json(
                run_dir / "state.json",
                error=str(error),
                exception_type=type(error).__name__,
            )
    finally:
        if deadline_handle is not None:
            deadline_handle.cancel()
        stop.set()
        stop_task.cancel()
        await snapshot_task
        try:
            final_snapshot = create_snapshot(
                run_dir / "live/submission", run_dir / "snapshots"
            )
            _log_event(
                "final_snapshot_finished",
                snapshot_id=(final_snapshot or {}).get("snapshot_id"),
                tree_sha256=(final_snapshot or {}).get("tree_sha256"),
            )
        except Exception as error:
            _log_event(
                "final_snapshot_failed",
                level="error",
                exception_type=type(error).__name__,
                error=str(error),
            )
        update_json(
            run_dir / "state.json",
            status=status,
            finished_at=_now(),
            stop_reason=stop_reason["value"],
        )
        _log_event(
            "worker_finished",
            status=status,
            exit_code=exit_code,
            stop_reason=stop_reason["value"],
        )
    return exit_code


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m agent_runner.worker RUN_DIR")
    raise SystemExit(asyncio.run(run_worker(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()

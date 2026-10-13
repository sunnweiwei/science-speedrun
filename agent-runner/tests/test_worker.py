import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from harbor.trial.hooks import TrialEvent

from agent_runner.io import read_json, write_json_atomic
from agent_runner.worker import (
    COMPUTE_BUDGET_INSTRUCTION_MARKER,
    _prepare_compute_budget,
    run_worker,
)


class _SuccessfulTrial:
    def __init__(self) -> None:
        self.hooks = {}

    def add_hook(self, event, hook) -> None:
        self.hooks[event] = hook

    async def run(self):
        await self.hooks[TrialEvent.AGENT_START](None)
        return SimpleNamespace(exception_info=None)


def test_compute_budget_records_deadline_and_accelerator(tmp_path: Path) -> None:
    runtime_task = tmp_path / "runtime-task"
    runtime_task.mkdir()
    (runtime_task / "instruction.md").write_text("Train carefully.")
    deadline = datetime(2026, 9, 3, tzinfo=timezone.utc)
    config = {
        "runtime_task_dir": str(runtime_task),
        "compute_profile": "default",
        "duration_seconds": 86400.0,
        "gpus": 1,
        "gpu_types": ["RTX 5080"],
        "cpus": None,
        "memory_mb": None,
        "storage_mb": None,
        "harness_llm_provider": "openai",
        "harness_llm_model": "model",
        "harness_llm_max_model_calls_per_case": 20,
        "harness_llm_development_total_tokens": 20_000_000,
        "harness_llm_max_output_tokens_per_call": 65_536,
    }

    budget_path = _prepare_compute_budget(tmp_path, config, deadline)
    budget = read_json(budget_path)
    assert budget["wall_time_seconds"] == 86400.0
    assert budget["deadline_utc"] == "2026-09-03T00:00:00+00:00"
    assert budget["resources"]["gpu_count"] == 1
    assert budget["resources"]["gpu_types"] == ["RTX 5080"]

    # Preparing a resumed worker must not duplicate the final instruction.
    _prepare_compute_budget(tmp_path, config, deadline)
    instruction = (runtime_task / "instruction.md").read_text()
    assert instruction.count(COMPUTE_BUDGET_INSTRUCTION_MARKER) == 1


def test_worker_emits_structured_lifecycle_events(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "live/submission").mkdir(parents=True)
    (run_dir / "runtime-task").mkdir()
    (run_dir / "runtime-task/instruction.md").write_text("Develop a model.\n")
    write_json_atomic(
        run_dir / "run.json",
        {
            "agent": "codex",
            "model": "openai/test",
            "network": "allowlist",
            "allowed_hosts": ["api.example.com"],
            "compute_profile": "test",
            "gpus": 0,
            "gpu_types": [],
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": None,
            "harness_llm_provider": "openai",
            "harness_llm_model": "model",
            "harness_llm_max_model_calls_per_case": 20,
            "harness_llm_development_total_tokens": 20_000_000,
            "harness_llm_max_output_tokens_per_call": 65_536,
            "duration_seconds": None,
            "snapshot_interval": 0.01,
            "runtime_env_names": ["PUBLIC_SETTING"],
            "agent_bins": {},
            "agent_kwargs": {"secret": "must-not-be-logged"},
            "runtime_task_dir": str(run_dir / "runtime-task"),
        },
    )
    write_json_atomic(
        run_dir / "state.json",
        {"schema_version": "agent-runner-state/v1", "status": "starting"},
    )

    trial = _SuccessfulTrial()

    captured_config = None

    async def create_trial(config):
        nonlocal captured_config
        captured_config = config
        return trial

    snapshot = {
        "snapshot_id": "snapshot-1",
        "tree_sha256": "abc123",
    }
    monkeypatch.setattr("agent_runner.worker.Trial.create", create_trial)
    monkeypatch.setattr("agent_runner.worker.create_snapshot", lambda *a, **k: snapshot)

    assert asyncio.run(run_worker(run_dir)) == 0

    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    events = [record["event"] for record in records]
    assert events == [
        "worker_started",
        "configuration_loaded",
        "compute_budget_prepared",
        "trial_creating",
        "trial_created",
        "snapshot_published",
        "agent_started",
        "trial_completed",
        "final_snapshot_finished",
        "worker_finished",
    ]
    assert all(record["timestamp"].endswith("+00:00") for record in records)
    assert "must-not-be-logged" not in output
    assert read_json(run_dir / "state.json")["status"] == "completed"
    budget = read_json(run_dir / "campaign-budget.json")
    assert budget == {
        "schema_version": "agent-runner-compute-budget/v1",
        "compute_profile": "test",
        "wall_time_seconds": None,
        "deadline_utc": None,
        "resources": {
            "gpu_count": 0,
            "gpu_types": [],
            "cpu_count": 2,
            "memory_mb": 4096,
            "storage_mb": None,
        },
        "harness_llm": {
            "provider": "openai",
            "model": "model",
            "limits": {
                "model_calls_per_case": 20,
                "development_total_tokens": 20_000_000,
                "output_tokens_per_call": 65_536,
            },
        },
        "unspecified_limits_use_runtime_defaults": True,
    }
    instruction = (run_dir / "runtime-task/instruction.md").read_text()
    assert instruction.startswith("Develop a model.\n")
    assert instruction.count(COMPUTE_BUDGET_INSTRUCTION_MARKER) == 1
    assert instruction.rstrip().endswith(
        "Do not exceed the budget or rely on resources absent from the\nJSON file."
    )
    assert captured_config is not None
    budget_mount = next(
        mount
        for mount in captured_config.environment.mounts
        if mount["target"] == "/workspace/campaign-budget.json"
    )
    assert budget_mount["read_only"] is True

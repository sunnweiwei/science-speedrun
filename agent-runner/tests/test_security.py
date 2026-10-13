from pathlib import Path

import pytest

from agent_runner.security import validate_auth_file
from agent_runner.io import write_json_atomic
from agent_runner.session import (
    _compose_projects_for_trials,
    _process_record,
    list_runs,
    process_alive,
    stop_run,
)


def test_auth_file_must_be_private(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    auth.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        validate_auth_file(auth)
    auth.chmod(0o600)
    assert validate_auth_file(auth) == auth.resolve()


def test_process_identity_prevents_stale_pid_reuse() -> None:
    import os

    record = _process_record(os.getpid())
    assert process_alive(record["pid"], record["start_ticks"])
    assert not process_alive(record["pid"], record["start_ticks"] + 1)


def _session(root: Path, name: str, *, status: str, pid: int, started_at: str) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    write_json_atomic(
        run_dir / "run.json",
        {
            "agent": "codex",
            "model": "openai/test",
            "task_dir": "/tasks/example",
            "compute_profile": "mac",
            "network": "allowlist",
        },
    )
    write_json_atomic(
        run_dir / "state.json",
        {"status": status, "started_at": started_at},
    )
    write_json_atomic(run_dir / "pid.json", {"pid": pid, "start_ticks": -1})
    return run_dir


def test_list_runs_reports_live_stale_and_terminal_sessions(tmp_path: Path, monkeypatch):
    live = _session(
        tmp_path, "live", status="running", pid=101, started_at="2026-01-03T00:00:00Z"
    )
    stale = _session(
        tmp_path, "stale", status="running", pid=102, started_at="2026-01-02T00:00:00Z"
    )
    done = _session(
        tmp_path, "done", status="completed", pid=103, started_at="2026-01-01T00:00:00Z"
    )
    monkeypatch.setattr(
        "agent_runner.session.process_alive",
        lambda pid, _start_ticks: pid == 101,
    )

    sessions = list_runs(tmp_path)
    assert [item["run_dir"] for item in sessions] == [str(live), str(stale), str(done)]
    assert [item["status"] for item in sessions] == ["running", "stale", "completed"]
    assert sessions[0]["agent"] == "codex"
    assert sessions[0]["latest_snapshot_id"] is None
    assert list_runs(tmp_path, active_only=True) == [sessions[0]]


def test_compose_project_discovery_requires_exact_trial_prefix(monkeypatch):
    monkeypatch.setattr(
        "agent_runner.session._docker_output",
        lambda _arguments: [
            "development-owned__env",
            "development-owned__verifier",
            "development-owned-extra__env",
            "unrelated__env",
        ],
    )
    assert _compose_projects_for_trials({"development-owned"}) == [
        "development-owned__env",
        "development-owned__verifier",
    ]


def test_stop_stale_run_removes_compose_resources_and_finalizes_state(
    tmp_path: Path, monkeypatch
):
    run_dir = _session(
        tmp_path,
        "stale-cleanup",
        status="running",
        pid=404404,
        started_at="2026-01-04T00:00:00Z",
    )
    submission = run_dir / "live/submission"
    submission.mkdir(parents=True)
    (submission / "answer.txt").write_text("preserve me")
    trial_name = "development-owned"
    (run_dir / "harbor" / trial_name).mkdir(parents=True)
    removed = []
    monkeypatch.setattr("agent_runner.session.process_alive", lambda *_args: False)
    monkeypatch.setattr(
        "agent_runner.session._compose_projects_for_trials",
        lambda names: [trial_name + "__env"] if names == {trial_name} else [],
    )
    monkeypatch.setattr(
        "agent_runner.session._remove_compose_project",
        lambda project, *, timeout: removed.append((project, timeout)),
    )

    result = stop_run(run_dir, timeout=7)

    assert removed == [(trial_name + "__env", 7)]
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "stale_worker_cleanup"
    assert result["stale_cleanup_projects"] == [trial_name + "__env"]
    assert result["latest_snapshot"]["snapshot_id"]
    events = (run_dir / "runner.log").read_text()
    assert "stale_cleanup_started" in events
    assert "stale_cleanup_finished" in events


def test_stop_stale_run_tolerates_missing_pid_metadata(tmp_path: Path, monkeypatch):
    run_dir = _session(
        tmp_path,
        "missing-pid",
        status="starting",
        pid=505505,
        started_at="2026-01-05T00:00:00Z",
    )
    (run_dir / "pid.json").unlink()
    (run_dir / "live/submission").mkdir(parents=True)
    monkeypatch.setattr(
        "agent_runner.session._compose_projects_for_trials", lambda _names: []
    )

    result = stop_run(run_dir)

    assert result["status"] == "stopped"
    assert result["stop_reason"] == "stale_worker_cleanup"

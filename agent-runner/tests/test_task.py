from pathlib import Path
from types import SimpleNamespace

import toml
import pytest
import yaml

from agent_runner.session import start_run
from agent_runner.evaluate import evaluate_snapshot
from agent_runner.io import write_json_atomic
from agent_runner.task import (
    harness_llm_environment,
    resolve_compute_profile,
    stage_task,
    task_gpu_types,
)


def fixture_task(root: Path) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.0"\nartifacts = ["/workspace/submission"]\n'
        '[task]\nname = "test/task"\n'
        '[environment]\ngpus = 8\ngpu_types = ["RTX TEST"]\n'
        '[verifier]\nenvironment_mode = "separate"\n'
        "[verifier.environment]\ngpus = 8\n"
    )
    (task / "instruction.md").write_text("develop")
    (task / "environment/Dockerfile").write_text("FROM scratch\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\n")
    for path in (
        task / "environment/docker-compose.yaml",
        task / "tests/docker-compose.yaml",
    ):
        path.write_text("services:\n  main:\n    shm_size: 1gb\n")
    return task


def test_stage_task_adds_offline_policy_images_and_gpu_devices(tmp_path: Path) -> None:
    source = fixture_task(tmp_path)
    (source / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    build: .\n"
    )
    (source / "tests/docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n    build: .\n"
        "  grader:\n    build: .\n"
        "  database:\n    build: ./database\n"
    )
    staged = stage_task(
        source,
        tmp_path / "runtime",
        network="allowlist",
        allowed_hosts=["api.example.com"],
        gpus=8,
        development_image="dev:test",
        verifier_image="verifier:test",
    )
    config = toml.load(staged / "task.toml")
    assert config["environment"]["network_mode"] == "allowlist"
    assert config["agent"]["allowed_hosts"] == ["api.example.com"]
    assert config["verifier"]["network_mode"] == "no-network"
    assert config["verifier"]["environment"]["docker_image"] == "verifier:test"
    assert config["environment"]["docker_image"] == "dev:test"
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        compose = yaml.safe_load(path.read_text())
        device = compose["services"]["main"]["deploy"]["resources"]["reservations"][
            "devices"
        ][0]
        assert device == {"driver": "nvidia", "count": 8, "capabilities": ["gpu"]}
        main = compose["services"]["main"]
        expected_image = "dev:test" if path.parent.name == "environment" else "verifier:test"
        assert main["image"] == expected_image
        assert "build" not in main
    verifier_services = yaml.safe_load(
        (staged / "tests/docker-compose.yaml").read_text()
    )["services"]
    assert verifier_services["grader"]["image"] == "verifier:test"
    assert "build" not in verifier_services["grader"]
    assert verifier_services["database"]["build"] == "./database"
    assert "docker_image" not in toml.load(source / "task.toml")["environment"]


def test_no_network_uses_compose_none_for_both_sandboxes(tmp_path: Path) -> None:
    staged = stage_task(
        fixture_task(tmp_path),
        tmp_path / "runtime",
        network="no-network",
        allowed_hosts=[],
        gpus=0,
    )
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        assert (
            yaml.safe_load(path.read_text())["services"]["main"]["network_mode"]
            == "none"
        )


def test_verifier_allowlist_is_staged_without_explicit_compose_network(
    tmp_path: Path,
) -> None:
    staged = stage_task(
        fixture_task(tmp_path),
        tmp_path / "runtime",
        network="no-network",
        allowed_hosts=[],
        gpus=0,
        verifier_network="allowlist",
        verifier_allowed_hosts=["api.example.com"],
    )
    config = toml.load(staged / "task.toml")
    assert config["environment"]["network_mode"] == "no-network"
    assert config["verifier"]["environment"]["network_mode"] == "allowlist"
    assert config["verifier"]["environment"]["allowed_hosts"] == ["api.example.com"]
    verifier_main = yaml.safe_load(
        (staged / "tests/docker-compose.yaml").read_text()
    )["services"]["main"]
    assert "network_mode" not in verifier_main


def test_stage_creates_compose_overlays_when_task_has_none(tmp_path: Path) -> None:
    source = fixture_task(tmp_path)
    (source / "environment/docker-compose.yaml").unlink()
    (source / "tests/docker-compose.yaml").unlink()
    staged = stage_task(
        source,
        tmp_path / "runtime",
        network="no-network",
        allowed_hosts=[],
        gpus=2,
    )
    for path in (
        staged / "environment/docker-compose.yaml",
        staged / "tests/docker-compose.yaml",
    ):
        main = yaml.safe_load(path.read_text())["services"]["main"]
        assert main["network_mode"] == "none"
        assert main["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 2


def test_start_rejects_missing_agent_binary_before_launch(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Agent executable does not exist"):
        start_run(
            task_dir=fixture_task(tmp_path),
            run_dir=tmp_path / "run",
            data_dir=None,
            agent="codex",
            model="openai/test",
            agent_kwargs={},
            network="allowlist",
            allowed_hosts=["api.example.com"],
            gpus="0",
            snapshot_interval=1,
            development_image=None,
            verifier_image=None,
            auth_file=None,
            agent_bins={"codex": str(tmp_path / "missing")},
        )


def test_compute_profile_uses_task_default_and_validates_resources(tmp_path: Path) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        'schema_version = "agent-runner-compute-profiles/v2"\n'
        'default = "gpu"\n'
        '[profiles.gpu]\n'
        'gpus = "task"\n'
        'cpus = 8\n'
        'memory_mb = 32768\n'
        'harness_llm_provider = "gemini"\n'
        'harness_llm_model = "gemini-2.5-pro"\n'
        'harness_llm_max_model_calls_per_case = 20\n'
        'harness_llm_development_total_tokens = 20000000\n'
        'harness_llm_max_output_tokens_per_call = 65536\n'
        '[profiles.mac]\n'
        'gpus = 0\n'
        'development_image = "task-mac:latest"\n'
        'verifier_image = "task-mac:latest"\n'
    )

    assert resolve_compute_profile(task, None) == {
        "name": "gpu",
        "gpus": "task",
        "cpus": 8,
        "memory_mb": 32768,
        "harness_llm_provider": "gemini",
        "harness_llm_model": "gemini-2.5-pro",
        "harness_llm_max_model_calls_per_case": 20,
        "harness_llm_development_total_tokens": 20_000_000,
        "harness_llm_max_output_tokens_per_call": 65_536,
    }
    assert resolve_compute_profile(task, "mac") == {
        "name": "mac",
        "gpus": 0,
        "development_image": "task-mac:latest",
        "verifier_image": "task-mac:latest",
    }
    assert task_gpu_types(task) == ["RTX TEST"]


def test_compute_profile_is_optional_but_explicit_unknown_profile_fails(
    tmp_path: Path,
) -> None:
    task = fixture_task(tmp_path)
    assert resolve_compute_profile(task, None) == {"name": None}
    with pytest.raises(ValueError, match="Task has no compute profiles"):
        resolve_compute_profile(task, "mac")


def test_v1_non_llm_profile_remains_compatible(tmp_path: Path) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        'schema_version = "agent-runner-compute-profiles/v1"\n'
        'default = "cpu"\n[profiles.cpu]\ngpus = 0\ncpus = 2\n'
    )
    assert resolve_compute_profile(task, None) == {
        "name": "cpu", "gpus": 0, "cpus": 2
    }


@pytest.mark.parametrize(
    "schema,extra",
    [
        ("v1", 'harness_llm_provider = "openai"\nharness_llm_model = "model"\n'),
        ("v2", 'harness_llm_provider = "openai"\nharness_llm_model = "model"\n'),
        ("v2", 'harness_llm_max_model_calls_per_case = 20\n'),
    ],
)
def test_harness_profile_missing_or_partial_budget_fails_closed(
    tmp_path: Path, schema: str, extra: str
) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        f'schema_version = "agent-runner-compute-profiles/{schema}"\n'
        f'default = "bad"\n[profiles.bad]\n{extra}'
    )
    with pytest.raises(ValueError, match="require schema v2"):
        resolve_compute_profile(task, None)


@pytest.mark.parametrize("value", [0, -1, True, "20"])
def test_harness_budget_fields_require_positive_integers(
    tmp_path: Path, value: object
) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        'schema_version = "agent-runner-compute-profiles/v2"\n'
        'default = "bad"\n[profiles.bad]\n'
        'harness_llm_provider = "openai"\nharness_llm_model = "model"\n'
        f'harness_llm_max_model_calls_per_case = {toml.dumps({"x": value}).split("=", 1)[1].strip()}\n'
        'harness_llm_development_total_tokens = 100\n'
        'harness_llm_max_output_tokens_per_call = 100\n'
    )
    with pytest.raises(ValueError, match="positive integer"):
        resolve_compute_profile(task, None)


def test_evaluate_rejects_incomplete_persisted_harness_profile_before_output(
    tmp_path: Path,
) -> None:
    task = fixture_task(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(run / "run.json", {
        "task_dir": str(task),
        "compute_profile": "old",
        "gpus": 0,
        "harness_llm_provider": "openai",
        "harness_llm_model": "model",
    })
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_snapshot(run, output_dir=output)
    assert not output.exists()


def test_evaluate_reuses_saved_profile_and_explicit_override_replaces_budget(
    tmp_path: Path, monkeypatch
) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        'schema_version = "agent-runner-compute-profiles/v2"\n'
        'default = "new"\n[profiles.new]\ngpus = 0\n'
        'harness_llm_provider = "gemini"\nharness_llm_model = "new-model"\n'
        'harness_llm_max_model_calls_per_case = 7\n'
        'harness_llm_development_total_tokens = 7000\n'
        'harness_llm_max_output_tokens_per_call = 700\n'
    )
    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(run / "run.json", {
        "task_dir": str(task), "compute_profile": "saved", "gpus": 0,
        "harness_llm_provider": "openai", "harness_llm_model": "saved-model",
        "harness_llm_max_model_calls_per_case": 3,
        "harness_llm_development_total_tokens": 3000,
        "harness_llm_max_output_tokens_per_call": 300,
    })
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr("agent_runner.evaluate.resolve_snapshot", lambda *_: snapshot)
    captured = []
    async def fake_evaluate(*args):
        captured.append(args[5])
        return {"profile": args[5]}
    monkeypatch.setattr("agent_runner.evaluate._evaluate", fake_evaluate)

    evaluate_snapshot(run, output_dir=tmp_path / "saved-output")
    evaluate_snapshot(
        run, output_dir=tmp_path / "new-output", compute_profile="new"
    )
    assert captured[0]["harness_llm_model"] == "saved-model"
    assert captured[0]["harness_llm_development_total_tokens"] == 3000
    assert captured[1]["harness_llm_model"] == "new-model"
    assert captured[1]["harness_llm_development_total_tokens"] == 7000
    assert harness_llm_environment(captured[1], None)[
        "HARNESS_LLM_TOTAL_TOKENS_LIMIT"
    ] == "unlimited"


def test_start_resolves_profile_and_cli_overrides(tmp_path: Path, monkeypatch) -> None:
    task = fixture_task(tmp_path)
    (task / "compute-profiles.toml").write_text(
        'schema_version = "agent-runner-compute-profiles/v2"\n'
        'default = "small"\n'
        '[profiles.small]\n'
        'gpus = 0\n'
        'cpus = 2\n'
        'memory_mb = 4096\n'
        'harness_llm_provider = "claude"\n'
        'harness_llm_model = "claude-sonnet"\n'
        'harness_llm_max_model_calls_per_case = 20\n'
        'harness_llm_development_total_tokens = 20000000\n'
        'harness_llm_max_output_tokens_per_call = 65536\n'
        'development_image = "profile-dev:test"\n'
        'verifier_image = "profile-verifier:test"\n'
    )
    launched = {}
    def popen(*args, **kwargs):
        launched.update(kwargs)
        return SimpleNamespace(pid=999999)
    monkeypatch.setattr("agent_runner.session.subprocess.Popen", popen)

    record = start_run(
        task_dir=task,
        run_dir=tmp_path / "run",
        data_dir=None,
        agent="codex",
        model="openai/test",
        agent_kwargs={},
        network="no-network",
        allowed_hosts=[],
        gpus=None,
        snapshot_interval=1,
        development_image="cli-dev:test",
        verifier_image=None,
        auth_file=None,
        cpus=4,
    )

    assert record["compute_profile"] == "small"
    assert record["gpus"] == 0
    assert record["gpu_types"] == []
    assert record["cpus"] == 4
    assert record["memory_mb"] == 4096
    assert record["development_image"] == "cli-dev:test"
    assert record["verifier_image"] == "profile-verifier:test"
    assert record["harness_llm_provider"] == "claude"
    assert record["harness_llm_model"] == "claude-sonnet"
    assert record["harness_llm_max_model_calls_per_case"] == 20
    assert record["harness_llm_development_total_tokens"] == 20_000_000
    assert record["harness_llm_max_output_tokens_per_call"] == 65_536
    assert launched["env"]["HARNESS_LLM_MAX_MODEL_CALLS_PER_CASE"] == "20"
    assert launched["env"]["HARNESS_LLM_TOTAL_TOKENS_LIMIT"] == "20000000"
    assert launched["env"]["HARNESS_LLM_MAX_OUTPUT_TOKENS_PER_CALL"] == "65536"

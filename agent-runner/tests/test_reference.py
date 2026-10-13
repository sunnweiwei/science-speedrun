import json
import os
from pathlib import Path
import shutil

import pytest
from types import SimpleNamespace
import yaml

from agent_runner.reference import (
    HARNESS_GATEWAY_HOST,
    expected_allowlist,
    load_reference_policy,
    provider_endpoint_hostname,
    with_harness_gateway_host,
)
from agent_runner.session import start_run


ROOT = Path(__file__).resolve().parents[2]
BIOMED_TASK = ROOT / "ai4sci-tasks" / "biomedicine-deep-research"


def materialized_reference(tmp_path: Path) -> tuple[Path, Path, dict]:
    task = tmp_path / "task"
    source = BIOMED_TASK / "environment/data/source_allowlist_manifest.json"
    destination = task / "environment/data/source_allowlist_manifest.json"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'schema_version = "1.0"\nartifacts = ["/workspace/submission"]\n'
        '[task]\nname = "test/biomed"\n[environment]\ngpus = 0\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    (task / "instruction.md").write_text("develop")
    (task / "tests/test.sh").write_text("#!/bin/sh\n")
    audit = json.loads(source.read_text())
    runtime = expected_allowlist(audit)
    data = tmp_path / "data"
    reference = data / "public/reference"
    reference.mkdir(parents=True)
    path = reference / "source-allowlist.json"
    path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    riskcalcs = reference / "riskcalcs.json"
    riskcalcs.touch()
    os.truncate(riskcalcs, audit["riskcalcs"]["size"])
    for view in ("development", "verifier"):
        directory = data / "public" / view
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "source_allowlist_sha256": digest,
        }))
    return task, data, runtime


def _mock_riskcalcs_hash(monkeypatch) -> None:
    import agent_runner.reference as reference
    original = reference._sha256
    monkeypatch.setattr(reference, "_sha256", lambda path: (
        "00a7a0089afffb66f2f32903bad94a5a2ea842841defb2d78e0686b0a5eb9ab9"
        if path.name == "riskcalcs.json" else original(path)
    ))


def test_reference_policy_validates_revision_hosts_and_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    task, data, runtime = materialized_reference(tmp_path)
    _mock_riskcalcs_hash(monkeypatch)
    policy = load_reference_policy(task, data)
    assert policy is not None
    assert policy["hosts"] == sorted(policy["hosts"])
    assert policy["hosts"] == runtime["network"]["allowed_hosts"]
    assert len(policy["hosts"]) == 22


@pytest.mark.parametrize("mutation", ["runtime", "hash", "missing"])
def test_reference_policy_fails_closed(tmp_path: Path, mutation: str, monkeypatch) -> None:
    task, data, _ = materialized_reference(tmp_path)
    _mock_riskcalcs_hash(monkeypatch)
    if mutation == "runtime":
        path = data / "public/reference/source-allowlist.json"
        value = json.loads(path.read_text())
        value["network"]["allowed_hosts"].pop()
        path.write_text(json.dumps(value))
    elif mutation == "hash":
        path = data / "public/verifier/manifest.json"
        path.write_text(json.dumps({"source_allowlist_sha256": "0" * 64}))
    else:
        (data / "public/reference/source-allowlist.json").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_reference_policy(task, data)


def test_provider_endpoints_are_exact_and_custom_urls_are_validated() -> None:
    online = {"BIOMED_GATEWAY_ONLINE": "1"}
    assert provider_endpoint_hostname("openai", online) == "api.openai.com"
    assert provider_endpoint_hostname("gemini", online) == "generativelanguage.googleapis.com"
    assert provider_endpoint_hostname("claude", online) == "api.anthropic.com"
    assert provider_endpoint_hostname("azure", {
        **online, "AZURE_API_BASE": "https://example-resource.openai.azure.com/openai",
    }) == "example-resource.openai.azure.com"
    with pytest.raises(ValueError):
        provider_endpoint_hostname("openai", {
            **online, "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        })


def test_internal_gateway_host_is_added_without_changing_biomedical_hosts() -> None:
    biomedical = ["api.ncbi.nlm.nih.gov", "www.ebi.ac.uk"]
    effective = with_harness_gateway_host(biomedical)
    assert effective == sorted(biomedical + [HARNESS_GATEWAY_HOST])
    assert biomedical == ["api.ncbi.nlm.nih.gov", "www.ebi.ac.uk"]


def test_start_merges_control_plane_and_biomedical_hosts(tmp_path: Path, monkeypatch) -> None:
    task, data, runtime = materialized_reference(tmp_path)
    _mock_riskcalcs_hash(monkeypatch)
    monkeypatch.setattr("agent_runner.session.subprocess.Popen",
                        lambda *args, **kwargs: SimpleNamespace(pid=999999))
    record = start_run(
        task_dir=task,
        run_dir=tmp_path / "run",
        data_dir=data,
        agent="codex",
        model="openai/test",
        agent_kwargs={},
        network="allowlist",
        allowed_hosts=["api.openai.com"],
        gpus="0",
        snapshot_interval=1,
        development_image=None,
        verifier_image=None,
        auth_file=None,
    )
    assert record["allowed_hosts"] == sorted(
        runtime["network"]["allowed_hosts"] + ["api.openai.com"]
    )
    assert record["biomedical_source_hosts"] == runtime["network"]["allowed_hosts"]
    assert record["biomedical_source_allowlist_sha256"]


def test_start_rejects_non_control_plane_extra_host(tmp_path: Path, monkeypatch) -> None:
    task, data, _ = materialized_reference(tmp_path)
    _mock_riskcalcs_hash(monkeypatch)
    with pytest.raises(ValueError, match="limited to the Codex control plane"):
        start_run(
            task_dir=task,
            run_dir=tmp_path / "run",
            data_dir=data,
            agent="codex",
            model="openai/test",
            agent_kwargs={},
            network="allowlist",
            allowed_hosts=["search.example.com"],
            gpus="0",
            snapshot_interval=1,
            development_image=None,
            verifier_image=None,
            auth_file=None,
        )


def test_biomedical_compose_uses_two_controlled_namespaces_and_offline_grader() -> None:
    for relative in ("environment/docker-compose.yaml", "tests/docker-compose.yaml"):
        compose = yaml.safe_load((BIOMED_TASK / relative).read_text())
        services = compose["services"]
        assert "networks" not in services["main"]
        assert "network_mode" not in services["main"]
        assert services["research-gateway"]["network_mode"] == \
            "service:provider-egress-control"
        assert "BIOMED_PROVIDER_ENDPOINT_HOST" in \
            services["research-gateway"]["environment"]
        provider = services["provider-egress-control"]
        assert provider["environment"]["EGRESS_CONTROL_INITIAL_NETWORK_MODE"] == \
            "allowlist"
        assert "BIOMED_PROVIDER_ENDPOINT_HOST" in \
            provider["environment"]["EGRESS_CONTROL_INITIAL_ALLOWED_HOSTS"]
    verifier = yaml.safe_load(
        (BIOMED_TASK / "tests/docker-compose.yaml").read_text()
    )["services"]
    assert verifier["grader"]["network_mode"] == "none"

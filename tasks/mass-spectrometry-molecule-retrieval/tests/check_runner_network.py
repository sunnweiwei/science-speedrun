#!/usr/bin/env python3
"""Check this task against real runner staging and Harbor egress selection.

This optional deployment check needs agent-runner, Harbor and PyYAML. It does
not start containers, contact a model API or certify runtime network isolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner-source", type=Path,
        help="Optional agent-runner checkout; otherwise use the installed package",
    )
    args = parser.parse_args()
    if args.runner_source:
        source = args.runner_source.resolve() / "src"
        if not (source / "agent_runner/task.py").is_file():
            parser.error("--runner-source must contain src/agent_runner/task.py")
        sys.path.insert(0, str(source))

    import yaml
    from agent_runner.task import stage_task
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import TaskConfig

    task = Path(__file__).resolve().parents[1]
    hosts = ["api.openai.com", "auth.openai.com", "chatgpt.com"]

    def config(root: Path) -> dict:
        value = tomllib.loads((root / "task.toml").read_text())
        TaskConfig.model_validate(value)
        return value

    def routed_services(compose: Path) -> list[str]:
        # Invoke Harbor's implementation without constructing a Docker runtime.
        context = SimpleNamespace(
            _environment_docker_compose_path=compose,
            extra_docker_compose_paths=[],
            _EGRESS_CONTROL_SERVICE_NAME=DockerEnvironment._EGRESS_CONTROL_SERVICE_NAME,
        )
        return DockerEnvironment._egress_controlled_service_names(context)

    original = config(task)
    assert original["environment"]["network_mode"] == "no-network"
    assert original["verifier"]["environment"]["network_mode"] == "no-network"
    cases = []
    with tempfile.TemporaryDirectory(prefix="massspecgym-runner-network-") as tmp:
        root = Path(tmp)
        for mode in ("allowlist", "no-network"):
            for gpus in (0, 1, 8):
                staged = stage_task(
                    task, root / f"{mode}-{gpus}", network=mode,
                    allowed_hosts=hosts if mode == "allowlist" else [], gpus=gpus,
                )
                resolved = config(staged)
                for table in ("environment", "agent"):
                    assert resolved[table]["network_mode"] == mode
                    assert resolved[table].get("allowed_hosts", []) == (
                        hosts if mode == "allowlist" else []
                    )
                verifier = resolved["verifier"]
                for policy in (verifier, verifier["environment"]):
                    assert policy["network_mode"] == "no-network"
                    assert not policy.get("allowed_hosts")
                development = staged / "environment/docker-compose.yaml"
                expected = ["main"] if mode == "allowlist" else []
                assert routed_services(development) == expected
                if mode == "no-network":
                    dev = yaml.safe_load(development.read_text())["services"]["main"]
                    assert dev["network_mode"] == "none"
                final_compose = staged / "tests/docker-compose.yaml"
                final = yaml.safe_load(final_compose.read_text())["services"]
                for name in ("main", "grader"):
                    assert final[name]["network_mode"] == "none"
                assert routed_services(final_compose) == []
                cases.append({"network": mode, "gpus": gpus, "status": "passed"})

        # Reproduce the old failure through the same runner, proving this check
        # observes the Compose conflict rather than only reading patched TOML.
        legacy = root / "legacy"
        shutil.copytree(task, legacy, symlinks=True)
        compose_path = legacy / "environment/docker-compose.yaml"
        compose = yaml.safe_load(compose_path.read_text())
        compose["services"]["main"]["network_mode"] = "none"
        compose_path.write_text(yaml.safe_dump(compose))
        staged = stage_task(
            legacy, root / "legacy-staged", network="allowlist",
            allowed_hosts=hosts, gpus=0,
        )
        assert config(staged)["environment"]["network_mode"] == "allowlist"
        assert routed_services(staged / "environment/docker-compose.yaml") == []

    print(json.dumps({
        "status": "passed", "profiles": cases,
        "legacy_compose_conflict_reproduced": True,
        "verifier_offline_in_all_profiles": True,
        "scope": "Real runner staging and Harbor service selection; no containers or API calls",
    }, indent=2))


if __name__ == "__main__":
    main()

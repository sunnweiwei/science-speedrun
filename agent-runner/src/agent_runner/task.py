from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Literal

import toml
import yaml


NetworkMode = Literal["task", "public", "no-network", "allowlist"]
PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PROFILE_FIELDS = {
    "gpus",
    "cpus",
    "memory_mb",
    "storage_mb",
    "development_image",
    "verifier_image",
    "harness_llm_provider",
    "harness_llm_model",
    "harness_llm_max_model_calls_per_case",
    "harness_llm_development_total_tokens",
    "harness_llm_max_output_tokens_per_call",
}

HARNESS_LLM_FIELDS = (
    "harness_llm_provider",
    "harness_llm_model",
    "harness_llm_max_model_calls_per_case",
    "harness_llm_development_total_tokens",
    "harness_llm_max_output_tokens_per_call",
)


def validate_resolved_harness_llm_profile(profile: dict) -> None:
    """Fail closed when persisted profile metadata is incomplete or malformed."""
    present = [field for field in HARNESS_LLM_FIELDS if profile.get(field) is not None]
    if not present:
        return
    if len(present) != len(HARNESS_LLM_FIELDS):
        raise ValueError("Persisted harness LLM profile is incomplete")
    for field in ("harness_llm_provider", "harness_llm_model"):
        if not isinstance(profile[field], str) or not profile[field].strip():
            raise ValueError(f"Persisted {field} must be a non-empty string")
    for field in HARNESS_LLM_FIELDS[2:]:
        _positive_integer(profile[field], field)


def harness_llm_environment(
    profile: dict, total_tokens_limit: int | None
) -> dict[str, str]:
    validate_resolved_harness_llm_profile(profile)
    if not profile.get("harness_llm_provider"):
        return {}
    if total_tokens_limit is not None:
        _positive_integer(total_tokens_limit, "total_tokens_limit")
    return {
        "HARNESS_LLM_PROFILE": str(profile["name"]),
        "HARNESS_LLM_PROVIDER": profile["harness_llm_provider"],
        "HARNESS_LLM_MODEL": profile["harness_llm_model"],
        "HARNESS_LLM_MAX_MODEL_CALLS_PER_CASE": str(
            profile["harness_llm_max_model_calls_per_case"]
        ),
        "HARNESS_LLM_TOTAL_TOKENS_LIMIT": (
            "unlimited" if total_tokens_limit is None else str(total_tokens_limit)
        ),
        "HARNESS_LLM_MAX_OUTPUT_TOKENS_PER_CALL": str(
            profile["harness_llm_max_output_tokens_per_call"]
        ),
    }


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Compute profile {field} must be a positive integer")
    return value


def resolve_compute_profile(task_dir: Path, requested: str | None) -> dict:
    """Resolve a task-owned compute profile without changing the source task."""
    task_dir = validate_task(task_dir)
    path = task_dir / "compute-profiles.toml"
    if not path.is_file():
        if requested is not None:
            raise ValueError(f"Task has no compute profiles: {path}")
        return {"name": None}

    document = toml.load(path)
    schema_version = document.get("schema_version")
    if schema_version not in {
        "agent-runner-compute-profiles/v1",
        "agent-runner-compute-profiles/v2",
    }:
        raise ValueError(f"Unsupported compute profile schema: {path}")
    name = requested if requested is not None else document.get("default")
    if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name):
        raise ValueError(f"Invalid or missing default compute profile in {path}")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ValueError(f"Unknown compute profile {name!r} in {path}")
    raw = profiles[name]
    if not isinstance(raw, dict):
        raise ValueError(f"Compute profile {name!r} must be a table")
    unknown = set(raw) - PROFILE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown fields in compute profile {name!r}: {', '.join(sorted(unknown))}"
        )

    result = {"name": name}
    if "gpus" in raw:
        value = raw["gpus"]
        if value != "task" and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(
                "Compute profile gpus must be 'task' or a non-negative integer"
            )
        result["gpus"] = value
    for field in ("cpus", "memory_mb", "storage_mb"):
        if field in raw:
            result[field] = _positive_integer(raw[field], field)
    for field in ("development_image", "verifier_image"):
        if field in raw:
            value = raw[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Compute profile {field} must be a non-empty string")
            result[field] = value
    present = [field for field in HARNESS_LLM_FIELDS if field in raw]
    if present and (
        schema_version != "agent-runner-compute-profiles/v2"
        or len(present) != len(HARNESS_LLM_FIELDS)
    ):
        raise ValueError(
            "Harness LLM compute profiles require schema v2 and all provider, model, "
            "and budget fields"
        )
    for field in ("harness_llm_provider", "harness_llm_model"):
        if field in raw:
            value = raw[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Compute profile {field} must be a non-empty string")
            result[field] = value.strip()
    for field in (
        "harness_llm_max_model_calls_per_case",
        "harness_llm_development_total_tokens",
        "harness_llm_max_output_tokens_per_call",
    ):
        if field in raw:
            result[field] = _positive_integer(raw[field], field)
    validate_resolved_harness_llm_profile(result)
    return result


def validate_task(task_dir: Path) -> Path:
    task_dir = task_dir.expanduser().resolve()
    required = (
        "task.toml",
        "instruction.md",
        "tests/test.sh",
    )
    missing = [name for name in required if not (task_dir / name).is_file()]
    if missing:
        raise ValueError(f"Not a supported Harbor task; missing {', '.join(missing)}")
    config = toml.load(task_dir / "task.toml")
    artifact_sources = {
        artifact if isinstance(artifact, str) else artifact.get("source")
        for artifact in config.get("artifacts", [])
    }
    if "/workspace/submission" not in artifact_sources:
        raise ValueError(
            "Model-development tasks must declare /workspace/submission as a Harbor artifact"
        )
    return task_dir


def require_separate_verifier(task_dir: Path) -> None:
    config = toml.load(task_dir / "task.toml")
    if config.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError(
            "Independent evaluation requires verifier.environment_mode = 'separate'"
        )


def task_gpu_count(task_dir: Path) -> int:
    config = toml.load(task_dir / "task.toml")
    return int(config.get("environment", {}).get("gpus") or 0)


def task_gpu_types(task_dir: Path) -> list[str]:
    config = toml.load(task_dir / "task.toml")
    raw = config.get("environment", {}).get("gpu_types") or []
    if not isinstance(raw, list) or not all(
        isinstance(value, str) and value.strip() for value in raw
    ):
        raise ValueError("environment.gpu_types must be a list of non-empty strings")
    return raw


def _set_policy(table: dict, mode: str, allowed_hosts: list[str] | None = None) -> None:
    table["network_mode"] = mode
    if mode == "allowlist":
        table["allowed_hosts"] = list(allowed_hosts or [])
    else:
        table.pop("allowed_hosts", None)


def _patch_network(
    config: dict,
    mode: NetworkMode,
    allowed_hosts: list[str],
    verifier_mode: NetworkMode = "no-network",
    verifier_allowed_hosts: list[str] | None = None,
) -> None:
    if mode == "task" and verifier_mode == "task":
        return
    environment = config.setdefault("environment", {})
    agent = config.setdefault("agent", {})
    verifier = config.setdefault("verifier", {})
    verifier_environment = verifier.setdefault("environment", {})
    if mode == "allowlist":
        if not allowed_hosts:
            raise ValueError("allowlist network mode needs at least one --allow-host")
        _set_policy(environment, mode, allowed_hosts)
        _set_policy(agent, mode, allowed_hosts)
    elif mode != "task":
        _set_policy(environment, mode)
        _set_policy(agent, mode)
    if verifier_mode == "allowlist":
        hosts = list(verifier_allowed_hosts or [])
        if not hosts:
            raise ValueError("verifier allowlist network mode needs at least one host")
        _set_policy(verifier_environment, verifier_mode, hosts)
        _set_policy(verifier, verifier_mode, hosts)
    elif verifier_mode != "task":
        _set_policy(verifier_environment, verifier_mode)
        _set_policy(verifier, verifier_mode)


def _patch_gpu_compose(path: Path, gpus: int) -> None:
    if gpus <= 0:
        return
    compose = yaml.safe_load(path.read_text()) if path.is_file() else {}
    compose = compose or {}
    main = compose.setdefault("services", {}).setdefault("main", {})
    deploy = main.setdefault("deploy", {})
    resources = deploy.setdefault("resources", {})
    reservations = resources.setdefault("reservations", {})
    devices = reservations.setdefault("devices", [])
    devices[:] = [
        device
        for device in devices
        if not isinstance(device, dict) or device.get("driver") != "nvidia"
    ]
    devices.append({"driver": "nvidia", "count": gpus, "capabilities": ["gpu"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))


def _patch_no_network_compose(path: Path) -> None:
    compose = yaml.safe_load(path.read_text()) if path.is_file() else {}
    compose = compose or {}
    compose.setdefault("services", {}).setdefault("main", {})["network_mode"] = "none"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compose, sort_keys=False))


def _build_context(service: dict) -> str | None:
    build = service.get("build")
    if isinstance(build, str):
        return build
    if isinstance(build, dict) and isinstance(build.get("context"), str):
        return build["context"]
    return None


def _patch_prebuilt_compose(path: Path, image: str) -> None:
    """Apply a prebuilt main image to services sharing its build context."""
    if not path.is_file():
        return
    compose = yaml.safe_load(path.read_text()) or {}
    services = compose.setdefault("services", {})
    main = services.setdefault("main", {})
    main_context = _build_context(main)
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        if name == "main" or (
            main_context is not None and _build_context(service) == main_context
        ):
            service.pop("build", None)
            service["image"] = image
    path.write_text(yaml.safe_dump(compose, sort_keys=False))


def stage_task(
    source: Path,
    destination: Path,
    *,
    network: NetworkMode,
    allowed_hosts: list[str],
    gpus: int,
    development_image: str | None = None,
    verifier_image: str | None = None,
    verifier_network: NetworkMode = "no-network",
    verifier_allowed_hosts: list[str] | None = None,
) -> Path:
    source = validate_task(source)
    if destination.exists():
        raise FileExistsError(f"Runtime task already exists: {destination}")
    shutil.copytree(source, destination, symlinks=True)
    config_path = destination / "task.toml"
    config = toml.load(config_path)
    _patch_network(
        config,
        network,
        allowed_hosts,
        verifier_network,
        verifier_allowed_hosts,
    )
    if development_image:
        config.setdefault("environment", {})["docker_image"] = development_image
    if verifier_image:
        config.setdefault("verifier", {}).setdefault("environment", {})[
            "docker_image"
        ] = verifier_image
    config_path.write_text(toml.dumps(config))
    if development_image:
        _patch_prebuilt_compose(
            destination / "environment/docker-compose.yaml", development_image
        )
    if verifier_image:
        _patch_prebuilt_compose(
            destination / "tests/docker-compose.yaml", verifier_image
        )
    _patch_gpu_compose(destination / "environment/docker-compose.yaml", gpus)
    _patch_gpu_compose(destination / "tests/docker-compose.yaml", gpus)
    if network == "no-network":
        _patch_no_network_compose(destination / "environment/docker-compose.yaml")
    if verifier_network == "no-network":
        _patch_no_network_compose(destination / "tests/docker-compose.yaml")
    return destination

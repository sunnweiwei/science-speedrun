"""Register a 50M config without replacing any upstream Protenix module.

The model is the upstream ``Protenix`` class and starts from the upstream
``protenix_tiny_default_v0.5.0`` scaling recipe. Only configuration values are
changed. Training, losses, permutation matching, diffusion, confidence, and
inference remain in the pinned upstream repository.
"""

from __future__ import annotations

from copy import deepcopy
import sys
from typing import Any


MODEL_NAME = "protenix_tiny_50m_v1"
UPSTREAM_MODEL_NAME = "protenix_tiny_default_v0.5.0"
UPSTREAM_REPOSITORY = "https://github.com/bytedance/Protenix"
UPSTREAM_REVISION = "c3bfc365b3e1341a11935eddfe7bfdc308092147"
UPSTREAM_IMAGE = (
    "ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4"
)
EXPECTED_PARAMETER_RANGE = (48_000_000, 50_000_000)


# The official Tiny recipe already scales the base model by reducing the MSA,
# Pairformer, atom encoder/decoder, and diffusion-transformer block counts. We
# retain those official choices and scale only channel dimensions further.
CHANNEL_OVERRIDES: dict[str, Any] = {
    "c_s": 256,
    "c_z": 64,
    "c_token": 256,
    "c_s_inputs": 321,  # c_token + 32 + 32 + 1, as required upstream
    "c_atom": 96,
    "model": {
        "diffusion_module": {
            "c_token": 512,
        },
    },
}


def deep_update(target: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply config overrides using the upstream runner semantics."""
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def model_override() -> dict[str, Any]:
    """Return the upstream Tiny recipe plus the 50M channel scaling."""
    from configs.configs_model_type import model_configs

    config = deepcopy(model_configs[UPSTREAM_MODEL_NAME])
    return deep_update(config, CHANNEL_OVERRIDES)


def register_model() -> str:
    """Add this config to the upstream registry and return its model name."""
    from configs.configs_model_type import model_configs

    existing = model_configs.get(MODEL_NAME)
    expected = model_override()
    if existing is not None and existing != expected:
        raise RuntimeError(f"upstream registry already has a different {MODEL_NAME}")
    model_configs[MODEL_NAME] = expected
    return MODEL_NAME


def enforce_cli_model_name() -> None:
    """Force the wrapper to run this baseline rather than an upstream default."""
    for index, argument in enumerate(sys.argv[1:], 1):
        if argument == "--model_name":
            if index + 1 >= len(sys.argv) or sys.argv[index + 1] != MODEL_NAME:
                raise ValueError(f"--model_name must be {MODEL_NAME}")
            return
        if argument.startswith("--model_name="):
            if argument.split("=", 1)[1] != MODEL_NAME:
                raise ValueError(f"--model_name must be {MODEL_NAME}")
            return
    sys.argv[1:1] = ["--model_name", MODEL_NAME]

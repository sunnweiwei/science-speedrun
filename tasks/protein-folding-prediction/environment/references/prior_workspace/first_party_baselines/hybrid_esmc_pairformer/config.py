"""Register the project-owned ESMC-6B + Protenix 50M composition."""

from __future__ import annotations

from copy import deepcopy
import sys

from official_baselines.protenix.config import deep_update, model_override


MODEL_NAME = "hybrid_esmc6b_pairformer_50m_v1"
# Upstream Protenix treats any model name containing ``esm`` as a request to
# download its unrelated ESM2-3B inference cache. The hybrid supplies frozen
# ESMC features through the audited adapter instead, so deployment uses an
# otherwise identical alias that does not trigger that heuristic.
INFERENCE_MODEL_NAME = "hybrid_pairformer_50m_v1"
ESMC_MODEL = "biohub/ESMC-6B"
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
FEATURE_ID = "esmc6b-mixed256-b28d8ace-45b0fa5d-ptsdpa-v1"
EXPECTED_PARAMETER_RANGE = (48_700_000, 48_900_000)
HYBRID_OVERRIDES = {
    "esm": {
        "enable": True,
        "model_name": FEATURE_ID,
        "embedding_dim": 256,
    },
    "use_msa": False,
}


def hybrid_override() -> dict:
    return deep_update(deepcopy(model_override()), HYBRID_OVERRIDES)


def register_model(model_name: str = MODEL_NAME) -> str:
    from configs.configs_model_type import model_configs

    if model_name not in {MODEL_NAME, INFERENCE_MODEL_NAME}:
        raise ValueError(f"unsupported hybrid model registry name: {model_name}")
    expected = hybrid_override()
    existing = model_configs.get(model_name)
    if existing is not None and existing != expected:
        raise RuntimeError(f"upstream registry already has a different {model_name}")
    model_configs[model_name] = expected
    return model_name


def enforce_cli_model_name(model_name: str = MODEL_NAME) -> None:
    if model_name not in {MODEL_NAME, INFERENCE_MODEL_NAME}:
        raise ValueError(f"unsupported hybrid model registry name: {model_name}")
    for index, argument in enumerate(sys.argv[1:], 1):
        if argument == "--model_name":
            if index + 1 >= len(sys.argv) or sys.argv[index + 1] != model_name:
                raise ValueError(f"--model_name must be {model_name}")
            return
        if argument.startswith("--model_name="):
            if argument.split("=", 1)[1] != model_name:
                raise ValueError(f"--model_name must be {model_name}")
            return
    sys.argv[1:1] = ["--model_name", model_name]

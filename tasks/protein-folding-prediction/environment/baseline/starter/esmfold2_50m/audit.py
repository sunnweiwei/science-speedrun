#!/usr/bin/env python3
"""Instantiate and audit the from-scratch starter without loading weights."""

import json
from pathlib import Path

from .architecture import build_model, persistent_state_dict
from .config import EXPECTED_PARAMETER_RANGE, MODEL_NAME_50M
from .recipe import parameter_count


def main() -> None:
    model = build_model()
    weight_path = Path(__file__).resolve().parents[1] / "model.safetensors"
    weights_loaded = weight_path.is_file()
    if weights_loaded:
        from safetensors.torch import load_file

        model.load_state_dict(
            persistent_state_dict(load_file(str(weight_path), device="cpu")),
            strict=True,
        )
    count = parameter_count(model)
    if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
        raise AssertionError(f"parameter count {count:,} is outside 45M--55M")
    print(json.dumps({
        "architecture": MODEL_NAME_50M,
        "parameter_count": count,
        "range": list(EXPECTED_PARAMETER_RANGE),
        "weights_loaded": weights_loaded,
        "pretrained_parameters": False,
        "input_feature": "ESMC-6B final normalized hidden [L,2560]",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Runnable approximately-50M ESMFold2 reference for the FoldBench task."""

from .architecture import build_model, persistent_state_dict, training_forward
from .config import EXPECTED_PARAMETER_RANGE, MODEL_NAME_50M, scaled_config

__all__ = [
    "EXPECTED_PARAMETER_RANGE",
    "MODEL_NAME_50M",
    "build_model",
    "persistent_state_dict",
    "scaled_config",
    "training_forward",
]

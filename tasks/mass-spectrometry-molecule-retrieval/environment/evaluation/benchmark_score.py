"""Versioned benchmark normalization shared by development and trusted grading."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

BOUNDS = json.loads(Path(__file__).with_name("score_bounds.json").read_text())


def normalize(performance: float) -> dict[str, Any]:
    """Normalize Hit@1 against fixed reference anchors, preserving overperformance."""
    if isinstance(performance, bool) or not math.isfinite(performance) or not 0 <= performance <= 1:
        raise ValueError("Hit@1 must be a finite number in [0, 1]")
    lower, upper = BOUNDS["lower"], BOUNDS["upper"]
    if not 0 <= lower < upper <= 1:
        raise ValueError("invalid benchmark score anchors")
    unclipped = (performance - lower) / (upper - lower)
    return {
        "version": BOUNDS["version"], "metric": BOUNDS["metric"],
        "direction": BOUNDS["direction"], "raw_performance": performance,
        "lower_reference": lower, "upper_reference": upper,
        "score": min(1.0, max(0.0, unclipped)), "unclipped_score": unclipped,
        "saturation": "lower" if unclipped < 0 else "upper" if unclipped > 1 else None,
        "at_or_above_upper_reference": performance >= upper,
    }

"""8-bit storage for training targets.

Training targets dominate the release: at full scale (all 41,020 eligible train intervals) float32
dense tracks and float16 contact maps need about 21.5 MiB per example, about 860 GiB in total.
Stored as 8-bit codes with per-track ranges they need about 7.9 MiB.

- Dense tracks are quantized in asinh space, the space the metric uses. On real examples the
  round trip keeps a per-track Pearson of at least 0.9997 in that space (median 0.99999).
- Contact maps are quantized linearly per track. They are already log-scale.

Validation and test targets are never quantized: the grader and the agent's validation score
use full precision.
"""

from __future__ import annotations

import numpy as np

DENSE_ENCODING = "asinh-uint8/v1"
CONTACT_ENCODING = "linear-uint8/v1"
LEVELS = 255


def encode(values: np.ndarray, *, asinh: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column uint8 codes with float32 lo/hi so that decode(codes) ~= values."""
    x = np.asarray(values, dtype=np.float64)
    if asinh:
        x = np.arcsinh(x)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    codes = np.rint((x - lo) / span * LEVELS)
    return (np.clip(codes, 0, LEVELS).astype(np.uint8), lo.astype(np.float32),
            hi.astype(np.float32))


def decode(codes: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, asinh: bool) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    x = lo + np.asarray(codes, dtype=np.float32) * ((hi - lo) / LEVELS)
    return np.sinh(x) if asinh else x

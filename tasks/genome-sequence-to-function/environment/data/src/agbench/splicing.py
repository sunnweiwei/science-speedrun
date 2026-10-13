"""Shared strand-aware splice coordinate utilities.

Site-position rows follow AlphaGenome's upstream contract:
0 donor-positive, 1 acceptor-positive, 2 donor-negative,
3 acceptor-negative. Coordinates are zero-based relative to the 1 Mb input.
"""

from __future__ import annotations

import numpy as np


SITE_POSITION_ROWS = (
    "donor_positive",
    "acceptor_positive",
    "donor_negative",
    "acceptor_negative",
)


def junction_coordinates(
    positions: np.ndarray,
    row: np.ndarray | int,
    column: np.ndarray | int,
    track_half: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve junction tensor indices to strand-specific input coordinates."""

    positions = np.asarray(positions)
    if positions.shape != (4, 512):
        raise ValueError(f"positions must have shape (4, 512), got {positions.shape}")
    row = np.asarray(row, dtype=np.int64)
    column = np.asarray(column, dtype=np.int64)
    track_half = np.asarray(track_half, dtype=np.int64)
    if np.any((track_half < 0) | (track_half > 1)):
        raise ValueError("track_half must contain only 0 (positive) or 1 (negative)")
    donor = positions[2 * track_half, row]
    acceptor = positions[2 * track_half + 1, column]
    valid = (donor >= 0) & (acceptor >= 0)
    return donor, acceptor, valid

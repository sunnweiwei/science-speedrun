"""Public prediction shapes; no trusted splice coordinates are exposed."""

from __future__ import annotations

DENSE_TRACKS = {
    "atac": 167,
    "dnase": 305,
    "procap": 12,
    "cage": 546,
    "rna_seq": 667,
    "chip_tf": 1_617,
    "chip_histone": 1_116,
}

PREDICTION_SCHEMA = "genome-sequence-to-function-predictions/v4"
MAX_USAGE_SITES = 4_096
MAX_JUNCTION_PAIRS = 16_384
# Fixed-shape arrays per example; splice usage and junctions are sparse (see contract.md).
FIXED_SHAPES = {
    **{name: (1_536, tracks) for name, tracks in DENSE_TRACKS.items()},
    "splice_sites": (196_608, 4),
    "contact_maps": (131_328, 28),
}

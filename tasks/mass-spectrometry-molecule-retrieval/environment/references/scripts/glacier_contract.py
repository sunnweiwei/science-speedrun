"""Shared public-data boundary for the released MassSpecGym GLACIER model."""

import math


SUPPORTED_INSTRUMENTS = frozenset({"Orbitrap", "QTOF"})
SELECTION_RULE = (
    "GLACIER iff adduct=[M+H]+, collision_energy is finite, and "
    "instrument_type is Orbitrap or QTOF"
)


def is_released_domain(query: dict) -> bool:
    """Return whether all conditions observed in released training are public."""
    collision_energy = query.get("collision_energy")
    return (
        query.get("adduct") == "[M+H]+"
        and isinstance(collision_energy, (int, float))
        and math.isfinite(collision_energy)
        and query.get("instrument_type") in SUPPORTED_INSTRUMENTS
    )

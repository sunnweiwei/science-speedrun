"""Machine description, grid and actuator limits for the MAST-U-like FreeGSNKE tokamak.

The coil/passive/limiter/wall/probe pickles are vendored from the FreeGSNKE repository
(LGPL-3.0, commit recorded in ``machine_configs/FREEGSNKE_COMMIT``). Voltage and current
limits follow the FreeGSNKE pulse-design-tool example; they are organizer-chosen
provisional values, not MAST-U engineering data, except the solenoid current range,
which is MAST-U's published one.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = os.environ.get(
    "TOKAGYM_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data" / "machine_configs"))

# Order must match FreeGSNKE's ``tokamak.coil_names[:12]`` for the vendored description.
ACTIVE_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5", "P6"]
VERTICAL_COIL = "P6"

# Absolute voltage limits [V], from the FPDT example (P6 is not in that example; provisional).
VOLTAGE_LIMITS = {"Solenoid": 2000.0, "PX": 750.0, "D1": 750.0, "D2": 750.0, "D3": 750.0,
                  "Dp": 750.0, "D5": 750.0, "D6": 750.0, "D7": 750.0, "P4": 750.0,
                  "P5": 750.0, "P6": 750.0}

# Coil current limits [A] (min, max), from the FPDT systems category; P6 provisional. The solenoid
# uses MAST-U's full range, "from 55kA down to -55kA" (Fishpool et al., J. Nucl. Mater. 438, 2013);
# the FPDT value of +/-10 kA matches the divertor coils ("total currents are up to ~+/- 10kA"),
# and with it no controller can hold the plasma current for a 0.35 s episode (v0.2 check).
CURRENT_LIMITS = {"Solenoid": (-55000.0, 55000.0), "PX": (-7000.0, 7000.0),
                  "D1": (-9000.0, 9000.0), "D2": (-9000.0, 9000.0), "D3": (-9000.0, 9000.0),
                  "Dp": (-9000.0, 9000.0), "D5": (-9000.0, 9000.0), "D6": (-9000.0, 9000.0),
                  "D7": (-9000.0, 9000.0), "P4": (-10000.0, 0.0), "P5": (-10000.0, 0.0),
                  "P6": (-5000.0, 5000.0)}


@dataclass
class Grid:
    Rmin: float = 0.1
    Rmax: float = 2.0
    Zmin: float = -2.2
    Zmax: float = 2.2
    nx: int = 65
    ny: int = 65


@dataclass
class MachineSpec:
    name: str = "MAST-U-like"
    data_dir: str = DEFAULT_DATA_DIR
    grid: Grid = field(default_factory=Grid)
    active_coils: tuple = tuple(ACTIVE_COILS)

    @property
    def n_active(self):
        return len(self.active_coils)

    @property
    def voltage_limits(self):
        return np.array([VOLTAGE_LIMITS[c] for c in self.active_coils])

    @property
    def current_limits(self):
        return np.array([CURRENT_LIMITS[c] for c in self.active_coils])

    def path(self, part):
        return str(Path(self.data_dir) / "MAST-U" / f"MAST-U_like_{part}.pickle")


def load_machine(spec: MachineSpec):
    """Build the FreeGSNKE tokamak object (with magnetic probes) and check coil order."""
    from freegsnke import build_machine

    tokamak = build_machine.tokamak(
        active_coils_path=spec.path("active_coils"),
        passive_coils_path=spec.path("passive_coils"),
        limiter_path=spec.path("limiter"),
        wall_path=spec.path("wall"),
        magnetic_probe_path=spec.path("magnetic_probes"),
    )
    names = list(tokamak.coil_names[: spec.n_active])
    if names != list(spec.active_coils):
        raise RuntimeError(f"Unexpected active coil order {names}")
    return tokamak

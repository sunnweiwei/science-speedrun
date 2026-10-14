# Data: machine description and reference equilibria

This task has **no large dataset**. Everything the simulator needs is in Git:

| Path | Content | Source |
| --- | --- | --- |
| `machine_configs/MAST-U/MAST-U_like_{active_coils,passive_coils,limiter,wall,magnetic_probes}.pickle` | 12 active coils, 138 passive filaments, limiter, wall, 102 flux loops + 398 pickup coils | FreeGSNKE repository, commit in `machine_configs/FREEGSNKE_COMMIT` (LGPL-3.0, see `LICENSE-FreeGSNKE`) |
| `machine_configs/MAST-U/simple_{diverted,limited}_currents_PaxisIp.pk` | Coil currents of the two example equilibria | FreeGSNKE `examples/data` |
| Lao85 double-null currents | inline in `tokagym/cases.py` | FreeGSNKE pulse-design-tool example |

The description is "MAST-U-like": UKAEA's public approximation, not engineering
drawings. Voltage/current limits are organizer-chosen values documented in
`tokagym/machine.py`, except the solenoid's ±55 kA, which is MAST-U's published range.
The randomization ranges in `tokagym/env.py` and `tests/sealed/` are derived from FAIR-MAST
by the developer-side tools in `fair_mast/`; the task's `.dockerignore` keeps that folder out of the
images, because the sealed ranges are derived from it. No credentials, downloads or Hugging Face assets are
involved; `TOKAGYM_DATA_DIR` points at `machine_configs/` inside the images.

Reference equilibria are rebuilt deterministically from these inputs at evaluator
start (static Newton-Krylov solve, ~2 s each), so no equilibrium files are stored.

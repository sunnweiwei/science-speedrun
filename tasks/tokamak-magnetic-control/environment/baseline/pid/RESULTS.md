# PID controllers: results

## v0.3 (`tokamak-shape-metrics/v3`, sealed seeds 1001-1005)

Privileged PID, vertical set-point from the target equilibrium (`env.vertical_target_at`), 2026-10-05:

| Case | Outcomes (1001-1005) | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | kept, kept, success, solver failure, solver failure | 1.21 | 0.35 | 0.36 | 0.19 | 0.16 |
| mastu_high_ip_hold | kept, kept, success, solver failure, solver failure | 1.21 | 0.35 | 0.36 | 0.19 | 0.16 |
| mastu_ip_ramp | kept, kept, success, solver failure, solver failure | 1.38 | 0.45 | 0.42 | 0.24 | 0.07 |
| mastu_p4_shape_hold | solver void, success, success, plasma lost, solver failure | 0.58 | 0.30 | 0.42 | 0.35 | 0.18 |
| mastu_pressure_drop | kept, kept, success, solver failure, solver failure | 1.37 | 0.35 | 0.33 | 0.18 | 0.12 |
| mastu_sequence | solver failure, kept, kept, solver failure, plasma lost | 1.24 | 0.57 | 0.52 | 25.18 | 0.15 |
| mastu_vertical_shift | kept, kept, success, solver failure, solver failure | 1.29 | 0.37 | 0.38 | 0.20 | 0.13 |

Success 7/34 (one void), kept 19/34. Seeds 1004/1005 are lost within tens of steps in most cases:
the FreeGSNKE PCS vertical gains oscillate once the supply delays by one or two steps (see
`../shape_vc/vertical_design.py`). The deployable `../shape_vc/` scores 17/35 on the same seeds.

## v0.2 (sealed-v0.2, `tokamak-shape-metrics/v2`, seeds 1001-1002)

Privileged PID with X-point-height feedback (the v0.2 default of `pid_reference.py`):

| Case | Outcomes (1001, 1002) | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | kept, kept | 1.40, 1.84 | 0.26, 0.49 | 0.32, 0.47 | 0.18, 0.22 | 0.14, 0.21 |
| mastu_high_ip_hold | kept, kept | 1.40, 1.84 | 0.26, 0.49 | 0.32, 0.47 | 0.18, 0.22 | 0.14, 0.21 |
| mastu_ip_ramp | kept, kept | 1.64, 2.11 | 0.32, 0.61 | 0.29, 0.61 | 0.22, 0.29 | 0.03, 0.13 |
| mastu_p4_shape_hold | void (stall at step 47), **success** | -, 0.91 | -, 0.35 | -, 0.55 | -, 0.39 | -, 0.22 |
| mastu_pressure_drop | kept, kept | 1.74, 2.02 | 0.21, 0.52 | 0.23, 0.51 | 0.14, 0.24 | 0.09, 0.16 |
| mastu_vertical_shift | kept, kept | 2.60, 1.80 | 0.25, 0.53 | 0.33, 0.52 | 7.50, 9.54 | 0.11, 0.18 |
| mastu_sequence | solver_failure ('ny' bug, step 461), kept | -, 6.29 | -, 0.77 | -, 0.78 | -, 0.48 | -, 0.18 |

Steady-state means; success needs every column within tolerance (1 cm, 3%). Success fraction 1/13
(one void), kept 12/13. In the holds, the current ramp and the pressure drop the gaps, X-point height
and current are all within the MAST-U tolerances and only the 32-point RMS (1.4-2.1 cm) misses 1 cm.
The vertical shift misses on the X-point height and the sequence on the RMS (the PID controls four
boundary quantities, not the whole shape change). The sequence's seed-1001 failure is the FreeGSNKE
`ny` bug that `tokagym.env.patch_freegsnke` now fixes; that job started before the patch.
Local run, 2026-10-03, four workers, profile updates throttled, X-point height metric as in METRICS.md.

Without X-point-height feedback (`--no-zx-feedback`, the v0.1 controller) the same seeds kept the
plasma in the seed-1001 holds at 1.0-1.35 cm but lost control in seed 1002, where the X-point drifted
until the solver stalled. On visible seed 0 of `mastu_diverted_hold` the feedback took the shape error
from 5.63 to 0.61 cm (success). The deployable package was not rebuilt under v0.2.

## v0.1 history

These are the task author's sealed-table runs: seeds 1001–1003, nonlinear plant, last-half window.
The integrator checked the artifacts but did not re-run them.

Privileged PID / virtual circuits (simulator-side descriptors; not a submission):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] |
| --- | --- | --- | --- | --- |
| mastu_diverted_hold | 2/3 | 1.66 | 2.39 | 0.11 |
| mastu_high_ip_hold | 2/3 | 1.66 | 2.39 | 0.11 |
| mastu_ip_ramp | 0/3 | - | - | - |
| mastu_p4_shape_hold | 3/3 | 0.62 | 1.49 | 1.04 |
| mastu_pressure_drop | 2/3 | 0.68 | 0.88 | 0.13 |
| mastu_vertical_shift | 2/3 | 1.38 | 4.82 | 0.12 |

Deployable PID package (v5, from the observation only):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] |
| --- | --- | --- | --- | --- |
| mastu_diverted_hold | 2/3 | 4.76 | 4.43 | 0.68 |
| mastu_high_ip_hold | 0/3 | - | - | - |
| mastu_ip_ramp | 0/3 | - | - | - |
| mastu_p4_shape_hold | 0/3 | - | - | - |
| mastu_pressure_drop | 2/3 | 3.53 | 2.53 | 0.76 |
| mastu_vertical_shift | 2/3 | 4.42 | 9.79 | 0.72 |

Most infeasible episodes are late-episode solver stalls, which count as infeasible for whoever is
evaluated. The deployable controller stalls the solver far more often (11/18 episodes) than the
privileged one (6/18). The remaining weakness is the vertical loop driven by a noisy estimated
current centroid.

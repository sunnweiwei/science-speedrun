# Baseline results under earlier task versions (maintainers)

Results recorded under `tokagym-mastu/v0.1`/`v0.2` and their metric versions, moved out of the agent
image: they were scored on an earlier plant and score and are not comparable with v0.3.

## From `environment/baseline/pid/RESULTS.md`

## v0.2 (`tokamak-shape-metrics/v2`, historical 2-seed check)

Privileged PID with X-point-height feedback (the v0.2 default of `pid_reference.py`):

| Case | Aggregate outcomes (2 episodes) | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 2 kept | 1.62 | 0.38 | 0.40 | 0.20 | 0.18 |
| mastu_high_ip_hold | 2 kept | 1.62 | 0.38 | 0.40 | 0.20 | 0.18 |
| mastu_ip_ramp | 2 kept | 1.88 | 0.47 | 0.45 | 0.26 | 0.08 |
| mastu_p4_shape_hold | 1 **success**, 1 void | 0.91 | 0.35 | 0.55 | 0.39 | 0.22 |
| mastu_pressure_drop | 2 kept | 1.88 | 0.37 | 0.37 | 0.19 | 0.13 |
| mastu_vertical_shift | 2 kept | 2.20 | 0.39 | 0.43 | 8.52 | 0.15 |
| mastu_sequence | 1 kept, 1 solver_failure | 6.29 | 0.77 | 0.78 | 0.48 | 0.18 |

Steady-state means; success needs every column within tolerance (1 cm, 3%). Success fraction 1/13
(one void), kept 12/13. In the holds, the current ramp and the pressure drop the gaps, X-point height
and current are all within the MAST-U tolerances and only the 32-point RMS (1.4-2.1 cm) misses 1 cm.
The vertical shift misses on the X-point height and the sequence on the RMS (the PID controls four
boundary quantities, not the whole shape change). The sequence failure is the FreeGSNKE
`ny` bug that `tokagym.env.patch_freegsnke` now fixes; that job started before the patch.
Local run, 2026-10-03, four workers, profile updates throttled, X-point height metric as in METRICS.md.

Without X-point-height feedback (`--no-zx-feedback`, the v0.1 controller) the same check kept the
plasma in one hold draw at 1.0-1.35 cm but lost control in the other draw, where the X-point drifted
until the solver stalled. On visible seed 0 of `mastu_diverted_hold` the feedback took the shape error
from 5.63 to 0.61 cm (success). The deployable package was not rebuilt under v0.2.

## v0.1 history

These are the task author's sealed-table runs: 3 blind seeds, nonlinear plant, last-half window.
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

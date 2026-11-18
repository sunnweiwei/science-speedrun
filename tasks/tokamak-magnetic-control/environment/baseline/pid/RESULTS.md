# PID controllers: results

The tables below use `tokamak-shape-metrics/v3` (35 sealed episodes, void episodes excluded). The privileged
PID runs through `pid_reference.py`, not the verifier's fallback ladder, so it was not re-scored under v4.

## v0.3 (`tokamak-shape-metrics/v3`, historical 5-seed sealed suite)

Privileged PID, vertical set-point from the target equilibrium (`env.vertical_target_at`), 2026-10-05:

| Case | Aggregate outcomes (5 episodes) | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 1 success, 2 kept, 2 solver failure | 1.21 | 0.35 | 0.36 | 0.19 | 0.16 |
| mastu_high_ip_hold | 1 success, 2 kept, 2 solver failure | 1.21 | 0.35 | 0.36 | 0.19 | 0.16 |
| mastu_ip_ramp | 1 success, 2 kept, 2 solver failure | 1.38 | 0.45 | 0.42 | 0.24 | 0.07 |
| mastu_p4_shape_hold | 2 success, 1 plasma lost, 1 solver failure, 1 void | 0.58 | 0.30 | 0.42 | 0.35 | 0.18 |
| mastu_pressure_drop | 1 success, 2 kept, 2 solver failure | 1.37 | 0.35 | 0.33 | 0.18 | 0.12 |
| mastu_sequence | 2 kept, 1 plasma lost, 2 solver failure | 1.24 | 0.57 | 0.52 | 25.18 | 0.15 |
| mastu_vertical_shift | 1 success, 2 kept, 2 solver failure | 1.29 | 0.37 | 0.38 | 0.20 | 0.13 |

Success 7/34 (one void), kept 19/34. Two supply-delayed perturbation draws are lost within tens of steps in most cases:
the FreeGSNKE PCS vertical gains oscillate once the supply delays by one or two steps (see
`../shape_vc/vertical_design.py`). The deployable `../shape_vc/` scores 17/35 on the same historical 5-seed suite.

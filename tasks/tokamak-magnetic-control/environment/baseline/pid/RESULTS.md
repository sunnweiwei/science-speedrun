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

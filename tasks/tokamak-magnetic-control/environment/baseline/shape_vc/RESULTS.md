# Full-shape virtual-circuit controller: results

## Development evaluation (visible randomization, seeds 0-2): the comparison for new candidates

This is `harbor_verify.py` with its defaults (all seven cases, visible randomization, seeds 0, 1, 2) on
`package/`, the protocol `/workspace/evaluation` runs. It has already been run, so compare a candidate
with these numbers instead of replaying the baseline:

```bash
python /workspace/evaluation/harbor_verify.py --submission /workspace/baseline/shape_vc/package \
    --output /workspace/research/shape_vc-dev --workers 21
```

**Score 15/19 = 0.789**, 19/19 non-void episodes kept, 2 void, policy computation 0.33 ms median.
Wall time 54 min with 21 workers (one per episode, `OMP_NUM_THREADS=4`, 96 CPUs) on a shared 208-core
host. A 700-step episode took 23-44 min there, and the 1,200-step sequence 51-54 min. An agent's replay of
the same suite with 2 workers on 32 CPUs took 3.1 h.

| Case | seed 0 | seed 1 | seed 2 | shape RMS [cm] | inner gap [cm] | outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | **success** | **success** | **success** | 0.42 | 0.28 | 0.47 | 0.25 | 0.17 |
| mastu_high_ip_hold | **success** | **success** | **success** | 0.36 | 0.34 | 0.41 | 0.26 | 0.17 |
| mastu_ip_ramp | **success** | **success** | **success** | 0.42 | 0.36 | 0.45 | 0.27 | 0.07 |
| mastu_p4_shape_hold | void (solver timeout, step 408) | void (solver timeout, step 492) | kept (Rin 1.06, Zx 1.10) | 0.95 | 1.06 | 0.46 | 1.10 | 0.19 |
| mastu_pressure_drop | **success** | **success** | **success** | 0.44 | 0.29 | 0.42 | 0.23 | 0.12 |
| mastu_sequence | **success** | kept (Rin 1.05) | **success** | 0.53 | 0.65 | 0.28 | 0.23 | 0.42 |
| mastu_vertical_shift | kept (Rin 1.23) | kept (Rin 1.39) | **success** | 0.75 | 1.07 | 0.27 | 0.67 | 0.13 |

How to read the table:
- Kept episodes name the steady-state quantities [cm] that missed the 1 cm tolerance.
- The physical columns are steady-state means over the non-void episodes.

Notes:
- **Voids vary between runs.** Both voids are FreeGSNKE step timeouts (180 s wall-clock) in P4. Void episodes are excluded from the score, and they vary from run to run on a loaded host: the same package scored 0.4848, 0.500 and 0.4857 in the sealed verifier (#117). Compare candidates on the same seeds, and repeat or inspect the per-case table before trusting a difference of one or two episodes.
- **Dev versus sealed.** The development plant is easier than the sealed one (0.789 here, 0.486 sealed). The remaining misses are the same kind as in the sealed run: inner gap and X-point height in P4, vertical shift and sequence.
- **Files.** `reproduction/dev-visible-seeds-0-2/` has the run's `results.json` (every episode: outcome, steady-state errors, latencies, setup draws, wall time), `grading.json` and `reward.json`.

## Sealed verifier, v0.3 (`tokagym-mastu/v0.3`, `tokamak-shape-metrics/v3`, seeds 1001-1005)

`tests/test.sh` on the package in `package/` (`w_x` 4, `k_zt` 150): **score 17/35 = 0.486**, 27/35
episodes kept, no void, policy computation 0.33 ms median (gate 0.5 ms).

| Case | 1001 | 1002 | 1003 | 1004 | 1005 |
| --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | **success** | **success** | **success** | lost (step 132) | **success** |
| mastu_high_ip_hold | kept (Rin 1.25) | **success** | **success** | lost (step 138) | **success** |
| mastu_ip_ramp | kept (Rin 1.53) | **success** | **success** | lost (step 133) | **success** |
| mastu_p4_shape_hold | kept (Rin 1.06) | kept (RMS 1.28, Rin 1.20, X-pt 1.97) | **success** | lost (step 122) | **success** |
| mastu_pressure_drop | kept (Rin 1.10) | **success** | **success** | lost (step 132) | **success** |
| mastu_vertical_shift | kept (RMS 1.03, Rin 2.18) | **success** | kept (Rin 1.24) | lost (step 132) | kept (Rin 1.70) |
| mastu_sequence | solver failure (step 511) | **success** | kept (Rin 1.64, X-pt 50) | lost (step 132) | kept (Rin 1.18, X-pt 1.38) |

Kept episodes name the steady-state quantities [cm] (Ip [%]) that missed the tolerance. Steady-state
means over the feasible episodes:

| Case | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 0.60 | 0.47 | 0.53 | 0.63 | 0.19 |
| mastu_high_ip_hold | 0.62 | 0.64 | 0.54 | 0.60 | 0.19 |
| mastu_ip_ramp | 0.64 | 0.73 | 0.50 | 0.61 | 0.08 |
| mastu_p4_shape_hold | 0.83 | 0.66 | 0.55 | 1.15 | 0.19 |
| mastu_pressure_drop | 0.60 | 0.53 | 0.44 | 0.61 | 0.13 |
| mastu_vertical_shift | 0.81 | 1.41 | 0.41 | 0.27 | 0.15 |
| mastu_sequence | 0.83 | 1.04 | 0.52 | 17.3 | 0.43 |

Reading:
- The remaining misses are mostly the inner gap (1.1-2.2 cm). The flux-loop integrator drift
  grows through the episode and the linear reconstruction does not model it, so the estimated
  inner separatrix drifts in seeds 1001 and 1003.
- Seed 1004 is lost in every case after 122-138 steps; the privileged PID loses it too.
- Since v0.3 the vertical moves target equilibria the machine can hold, and the controller moves
  its vertical set-point from the P6 part of the least-squares correction. The vertical shift and the
  sequence each have a success. Under v0.2 (rigidly shifted targets) no controller could meet the
  X-point tolerance there.

Comparison on the same sealed seeds (v0.3): hold-current 0/35 (all lost); privileged PID (simulator
descriptors, FreeGSNKE PCS gains) 7/34 success (one void), 19/34 kept; its holds miss on the 1.2-1.4 cm
shape RMS and it loses seeds 1004/1005 (vertical loop under supply delay).

## v0.2 history

Under v0.2 (rigid vertical shifts, X-point tolerance relative to psi_bndry) the same control law
without the vertical outer loop scored 11/34 (one void), 26/34 kept. Vertical shift and sequence had
no success there: X-point height errors were 7-50 cm, and for the sequence the shape RMS was 3.6-4.3 cm.

## Development plant (dev randomization, seeds 5001-5007, not the sealed seeds)

The same controller run on the simulator boundary instead of the estimator gives the upper bound
of the control law: 15/28 success, 23/28 kept (v0.2). Changes to the deployable package were checked
on 450- and 700-step proxies:
- the final estimator (all data) with X-point weight 4: 14-15 of 30 episodes with every error below 1 cm;
- the vertical outer loop (v0.3): with the simulator boundary, 10 of 12 vertical-shift, hold and P4
  episodes pass, against 6 of 12 without it.

Run 2026-10-05, 35 verifier workers on a 208-core host shared with the privileged-PID run, nonlinear
solver, FreeGSNKE 3.1.0.

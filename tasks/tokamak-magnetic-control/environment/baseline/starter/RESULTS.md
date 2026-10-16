# Starter: results

## v0.3: residual SAC on `shape_vc` (run 2026-10-05)

Recipe (the README's minimal loop, step 2):
- `train.py --residual-package /workspace/baseline/shape_vc/package --residual-scale 0.05
  --workers 64 --hours 5 --solver nonlinear --demo-episodes 1 --demo-min-episodes 32
  --critic-warmup 1000 --publish-every 24 --snapshot-every 24`;
- visible randomization, one GPU for the learner.

Training:
- 5 h, 311k simulator steps, 145k SAC updates.
- 64 base-controller demonstrations: 59 kept the plasma for the full horizon.
- 350 actor episodes with exploration noise: 337 kept.
- The residual stayed small, about 0.6-1% of the voltage limits.
- The publish-if-better rule published at episodes 96, 120 and 144. The training-window kept fraction went from 0.71 for the base to 0.99.
- The last published actor (144 episodes, 105k steps) is the package below.

Inference:
- The actor runs in float32 on 1-D vectors, about 0.06 ms; the whole package takes about 0.11 ms in process.
- Parameters: actor 285k, plus the base estimator's solved weights (106k).

`tests/test.sh`, all sealed seeds 1001-1005, v0.3:

| Case | Outcomes | Shape RMS [cm] | Inner gap [cm] | Outer gap [cm] | X-point height [cm] | Ip error [%] |
| --- | --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 3 success, 1 lost, 1 void | 0.60 | 0.32 | 0.57 | 0.56 | 0.21 |
| mastu_high_ip_hold | 3 success, 1 kept, 1 lost | 0.62 | 0.64 | 0.54 | 0.60 | 0.19 |
| mastu_ip_ramp | 3 success, 1 kept, 1 solver failure | 0.64 | 0.73 | 0.50 | 0.61 | 0.08 |
| mastu_p4_shape_hold | 2 success, 2 kept, 1 solver failure | 0.83 | 0.66 | 0.56 | 1.15 | 0.19 |
| mastu_pressure_drop | 3 success, 1 kept, 1 lost | 0.60 | 0.53 | 0.44 | 0.61 | 0.13 |
| mastu_vertical_shift | 1 success, 3 kept, 1 lost | 0.82 | 1.41 | 0.41 | 0.28 | 0.15 |
| mastu_sequence | 1 success, 2 kept, 1 lost, 1 solver failure | 0.83 | 1.04 | 0.52 | 17.3 | 0.43 |

**Score 16/34 = 0.471** (one void), 26/34 kept, policy computation 0.46 ms median (gate 0.5 ms;
35 verifier workers on a loaded host). The base alone scores 17/35 (`../shape_vc/RESULTS.md`).

The pipeline is verified end to end: training updates, publication, independent replay. The
residual does not improve on the base yet: per-case errors match `shape_vc` within a few mm,
because the learned residual stays within about 1% of the voltage limits. The RL improvement step
is open. The latency budget left for a larger actor is small on a loaded verifier.

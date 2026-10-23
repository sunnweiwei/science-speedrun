# Baselines

| baseline | what it is | results |
| --- | --- | --- |
| [`shape_vc/`](shape_vc/) | deployable full-shape virtual-circuit controller: linear magnetic reconstruction, least-squares virtual circuits on the 32 target points and X-points, delay-robust vertical PID; NumPy, classical | **metric v4: 48/105 success, 79/105 kept** (sealed seeds 1001-1015), see `shape_vc/RESULTS.md`; development suite (visible, seeds 0-2): 16/21 = 0.762 |
| [`starter/`](starter/) | residual SAC on a deployable base (`shape_vc`), or behaviour cloning from the privileged PID plus SAC; NumPy-deployable | v0.3 residual on `shape_vc`, metric v3 (35 episodes): 16/34 success, 26/34 kept (no gain over the base yet); not re-scored under v4, see `starter/RESULTS.md` |
| [`pid/`](pid/) | privileged PID reference (not a submission) and its deployable package | privileged v0.3, metric v3: 7/34 success, 19/34 kept (seeds 1001-1005, see `pid/RESULTS.md`); deployable package rebuilt from v0.2 demonstrations: 0/35 kept (coil limits from estimator bias, vertical loop unstable under supply delay) |
| [`hold_current/`](hold_current/) | `V = R I`, the no-control anchor | metric v4: 0/105 (all lost), see `hold_current/RESULTS.md` |

Each directory's `RESULTS.md` has per-case tables under the current plant and score
(`tokagym-mastu/v0.3`, `tokamak-shape-metrics/v3`).

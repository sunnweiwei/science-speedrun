# Baselines

| baseline | what it is | v0.3 |
| --- | --- | --- |
| [`shape_vc/`](shape_vc/) | deployable full-shape virtual-circuit controller: linear magnetic reconstruction, least-squares virtual circuits on the 32 target points and X-points, delay-robust vertical PID; NumPy, classical | **v0.3: 17/35 success, 27/35 kept** (seeds 1001-1005), see `shape_vc/RESULTS.md`; development suite (visible, seeds 0-2): 15/19 = 0.789, 2 void |
| [`starter/`](starter/) | residual SAC on a deployable base (`shape_vc`), or behaviour cloning from the privileged PID plus SAC; NumPy-deployable | v0.3 residual on `shape_vc`: 16/34 success, 26/34 kept (no gain over the base yet), see `starter/RESULTS.md` |
| [`pid/`](pid/) | privileged PID reference (not a submission) and its deployable package | privileged v0.3: 7/34 success, 19/34 kept (seeds 1001-1005, see `pid/RESULTS.md`); deployable package rebuilt from v0.2 demonstrations: 0/35 kept (coil limits from estimator bias, vertical loop unstable under supply delay) |
| [`hold_current/`](hold_current/) | `V = R I`, the no-control anchor | see `hold_current/RESULTS.md` |

Each directory's `RESULTS.md` has per-case tables under the current plant and score
(`tokagym-mastu/v0.3`, `tokamak-shape-metrics/v3`).

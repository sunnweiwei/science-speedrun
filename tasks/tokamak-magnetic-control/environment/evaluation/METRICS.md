# Fixed evaluation metrics: `tokamak-shape-metrics/v4`

Adopted 2026-10-22, replacing v3 of 2026-10-05 (v2: 2026-10-03, v1: 2026-09-20). Covers the seven
cases of `tokagym-mastu/v0.3`. v4 makes the score reproducible: every episode is scored (no void
episodes, one denominator for every package), numerical solver failures are retried before they
count (below), host-side failures are rerun instead of scored, and the latency gate is measured
after the replay on an otherwise idle evaluator. It adds the continuous `tolerance_time_fraction`,
and the sealed suite has 15 seeds per case (105 episodes) instead of 5. v3 changed two things
relative to v2: vertical moves target equilibria the machine can hold instead of a rigid shift,
and X-point candidates use a psi_N tolerance instead of one relative to the drifting boundary flux
(both under "Exact computation"). Changing a metric, window, tolerance or direction requires a new
version.
All values are computed by the trusted evaluator from the simulator state; the policy never
reports them.

## Score

`reward` (in `/logs/verifier/reward.json`) is the **fraction of all episodes that succeed**,
pooled over every case and sealed seed. An episode succeeds when the plasma is kept for the full
horizon and its steady-state errors are within the MAST-U controller requirement:

| Quantity (steady-state mean) | Tolerance | Source |
| --- | --- | --- |
| Boundary-shape RMS error | ≤ 1 cm | MAST-U: "small steady-state error (about 1 cm or less)" (Lvovskiy et al., PPCF 67, 2025) |
| Inner gap, outer gap, lower X-point height errors (each) | ≤ 1 cm | same; MAST-U controls these gaps (Anand et al., Nucl. Fusion 64, 2024) |
| Relative plasma-current error | ≤ 3% | no MAST-U tolerance is published; Degrave et al. (Nature 602, 2022) |

It is a measured quantity (a success rate under operational tolerances), so the task needs no
exception from the repository's single-reward rule. Every episode is in the denominator. An
invalid package scores 0: a policy error in any case, or a median per-step computation above one
control period (below).

`tolerance_time_fraction` is its continuous companion: per episode, the share of the steady-state
window (below) whose steps are all within the same tolerances; steps an episode did not reach count
as outside. It is averaged over all episodes. It moves when a controller improves by millimetres
that do not yet flip an episode.

## Steady-state window

The MAST-U team asks for a "fast settling response time (about 100 ms)" and did not request a
small transient error. The window therefore starts 100 ms after the last change of the case's
references (current or shape waveform, or the unannounced disturbance; the reset counts as a
change because the plasma starts off its targets) and runs to the end of the horizon. If that
leaves less than 50 ms, the window is the last 50 ms; with the v0.2 horizons every case keeps at
least 100 ms. `records[j]` is the state after step j+1, so the window is
`records[round(start / dt):]`. A window holding a non-finite error has no steady-state value and
does not succeed.

## Outcomes

| Outcome | Meaning | Scored |
| --- | --- | --- |
| `success` | full horizon, all steady-state errors within tolerance | yes |
| `kept` | full horizon, a tolerance missed | yes (not a success) |
| `plasma_lost` | `Ip < 20%` of target, magnetic axis more than 0.25 m from its reference in R or Z, no closed boundary, or boundary RMS error above 0.20 m | yes |
| `coil_limit` | an active-coil current left its limit (`tokagym/machine.py`); the episode ends, as machine protection would (Degrave et al. 2022) | yes |
| `solver_failure` | a FreeGSNKE exception, a non-finite state or an exhausted work budget that the fallback ladder could not recover, or an episode that stalled again when rerun with doubled budgets | yes (not a success) |

`within_tolerance_at_failure` in `results.json` records whether the mean of the last 10 steps before
a solver failure had shape RMS ≤ 1 cm and current error ≤ 3%. Until v3 such episodes were voided
and left the denominator; that let host load (a 180 s wall-clock budget per step) change both the
numerator and the denominator, and it rewarded a controller for stalling the solver.

## Numerical failures and reruns

Every retry is bounded and recorded in `results.json`; none changes the plant draw, and none depends
on the speed of the host.

- **Work budgets.** Each solve may evaluate FreeGSNKE's residuals (dynamic currents and flux, static
  Grad-Shafranov) a fixed number of times (`tokagym/env.py`): 20,000 per control step, per
  feedforward-hold step of the handover and per static solve; a normal step takes about 550
  (p99 about 2,600, about 3 ms each). A wall-clock budget would make the same step pass on an idle
  host and fail on a loaded one; a work budget gives the same answer everywhere.
- **Fallback ladder for one step.** When a control step raises, returns a non-finite state or
  exhausts its work budget, the evaluator resets the episode with the same seed and re-applies the
  recorded actions without the policy (the single-threaded simulator is deterministic; the replayed
  observation must equal the one the policy saw, bit for bit, or the run is an infrastructure
  error). It then retries the step at level 1 (150 instead of 50 Newton-Krylov iterations, four
  times the work budget) and, if that fails, at level 2 (this step only through FreeGSNKE's
  linearised model; the next step is nonlinear again). A step that converges within the level-0
  limits gives the same result at level 1. At most 3 failing steps per episode are retried, and the
  replays of one episode may total at most four times its horizon (so an episode costs at most about five times
  its length; on the sealed suite no episode of the classical baseline came close except one sequence episode that
  failed anyway); the next failure, or a step that fails at every level or finds no replay budget left, ends the episode as
  `solver_failure`.
  `fallback_steps` counts the steps that needed level 1 or 2.
- **Setup.** A static solve over its work budget is a non-converged draw and is redrawn from the
  seed, as before; a handover step that fails or exceeds its work budget ends the handover (the
  controller takes over the converged initial plasma), as before.
- **Host-side guards and reruns.** Wall-clock limits only catch hangs the work budget cannot see:
  1,800 s per solve, 2,400 s without any message from a worker, 6 h per episode. An episode they
  stop, a crashed worker or an evaluator error is rerun once in a fresh process with doubled
  wall-clock limits (`episode_reruns`). If the rerun stalls again, the episode is a
  `solver_failure`; if it crashes or errs again, the replay publishes no score (infrastructure
  error). A policy error is the package's fault and is never rerun.

## Per-case reported values

| Key (`CASE/...`) | Definition | Units | Direction |
| --- | --- | --- | --- |
| `success_fraction` | Fraction of episodes that succeed | 0..1 | higher |
| `feasible_fraction` | Fraction of episodes that ran the full horizon | 0..1 | higher |
| `tolerance_time_fraction` | Mean share of the steady-state window within every tolerance | 0..1 | higher |
| `solver_failure_episodes` | Episodes ended by an unrecovered solver failure | count | - |
| `shape_rms_cm` | Mean over feasible episodes of the steady-state boundary-shape RMS error | cm | lower |
| `rin_error_cm`, `rout_error_cm`, `zx_error_cm` | Same for the inner gap, outer gap and lower X-point height | cm | lower |
| `xpoint_error_cm` | Same for the X-point position error (capped at 50 cm per step) | cm | lower |
| `ip_error_pct` | Same for the relative plasma-current error | % | lower |

Top level: `reward`, `tolerance_time_fraction`, `policy_compute_ms` (the gate),
`policy_compute_ms_under_load` (diagnostic, measured during the replay), `episodes`,
`solver_failure_episodes`, `fallback_steps` and `episode_reruns`. A case with no feasible episode
publishes only its fractions; no error value is invented for it.

## Exact computation

- **Boundary-shape RMS error.** At each control step the evaluator extracts the closed
  normalized-flux surface psi_N = 0.995 (marching squares on the equilibrium grid, the contour
  enclosing the magnetic axis; a proxy for the last closed flux surface that stays well defined
  next to the X-point) and computes, for each of the 32 target control points, the Euclidean
  distance to the closed polyline; the RMS over the 32 points is the step value. Targets are
  sampled from the reference equilibrium's contour, equally spaced in arc length from the
  outboard midplane counter-clockwise. Every target is a blend of equilibria the machine can
  hold: for `mastu_sequence` the targets blend linearly into the second reference shape, and a
  vertical move blends toward the same plasma held higher by the vertical coil P6 alone (its
  current solved so that the current centroid sits at the case's `z_shift`). A rigid shift is
  not used: the shaping coils are up-down symmetric series pairs, so a double null moved
  vertically becomes slightly single-null (2 cm: 1.5 cm RMS from the rigid shift, lower X-point
  at psi_N 1.02; 10 cm: 5 cm, psi_N 1.09).
- **Gaps.** Inner and outer separatrix radii at the target axis height (FreeGSNKE
  `innerOuterSeparatrix`) and the height of the candidate X-point nearest to the target lower
  X-point (capped at 0.5 m, like the X-point error), against the same quantities of the
  (blended) target equilibria.
- **X-point error.** For each target X-point, the distance to the nearest candidate X-point
  (X-points with |psi_N - 1| <= 0.1; if none, all saddle points), capped at 0.5 m; a missing
  X-point counts as the cap. Mean over the target X-points. The tolerance is on the normalized
  flux: a tolerance relative to psi_bndry, which drifts with resistive flux consumption, dropped
  a lower X-point a few mm off the boundary during the episode. Target X-points are the X-point
  nearest the boundary flux below and above the axis within the same tolerance.
- **Plasma-current error.** `|Ip - Ip_target(t)| / Ip_target(t)` with `Ip` the simulator's total
  plasma current.
- **Latency gate.** Measured after the replay, when no episode is running: for every case, the
  observations the package received in the first 600 steps of its first-seed episode are sent
  again, in order, to a fresh package process (reset as in the episode) and to the null policy
  (`evaluation/null_policy/`, which parses the request and answers zero volts), alternating which
  answers first. The computation is the median over all these steps of (submission round trip −
  null round trip) and must not exceed one control period, 0.5 ms. Until v3 the same difference
  was taken during the replay, where the evaluator's own load moved packages near the limit across
  it between runs; that value is still reported as `policy_compute_ms_under_load`. Degrave et
  al.'s actor ran in 50 µs on TCV; the JSON protocol cannot resolve that, so the gate is one
  control period.

- **Controllable draws only.** A plant draw whose fastest vertical-instability timescale
  (`growth_rate` in the episode setup, in seconds) is below 2 ms, four control periods, is redrawn
  deterministically from the seed. The supply alone adds up to 1 ms of delay. Two of the five
  sealed seeds first drew 0.6-1.0 ms plasmas, which even the privileged PID lost within five steps.
  `controllability_redraws` records how many redraws a reset needed.

## Diagnostics (not scores)

`results.json` also records per episode the training-style return, the last-half means of v1,
policy and null latencies, setup draws (plasma resistivity, passive resistance scale, supply
delay, profile constraint, drifts, crashes, dead channels, handover steps) and wall time.

## Reporting contract

Compare candidates, the hold-current anchor and the PID reference on the same metric version,
randomization name and seed set. State the randomization name (`visible`, `none` or the sealed
file) in every table. Subset replays are development evidence, never suite scores.

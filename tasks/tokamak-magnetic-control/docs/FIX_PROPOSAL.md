# Fix proposal: realism and closeness to Degrave et al. (2022)

Status: proposal from the task verifier, 2026-10-02; implemented as `tokagym-mastu/v0.2` and `tokamak-shape-metrics/v2` on 2026-10-03 (see the next section), not yet committed.
Goals: (1) make the task useful for fusion research; (2) keep it close to Degrave et al., Nature 602 (2022), accepting harder components where they add realism.

## Implementation status (2026-10-03)

- Fix 1 is implemented in `tokagym/metrics.py`, `tokagym/env.py`, `evaluation/harbor_verify.py` and `evaluation/supervise.py`, with unit tests in `tests/test_metrics.py`.
- Episode success uses the MAST-U values: steady-state shape and inner gap, outer gap and X-point height each within 1 cm, plasma current within 3%, scored from 100 ms after the last reference change.
- Coil-current limits terminate the episode, stalls of a plasma within tolerance are void, more than 20% void episodes withhold the score, and `reward` is the success fraction.
- Enforcing the limits showed that the solenoid's provisional ±10 kA ended every privileged-PID episode after about 0.13 s; MAST-U's solenoid range is ±55 kA (Fishpool et al., J. Nucl. Mater. 438, 2013), and `tokagym/machine.py` now uses it.
- Fix 2 is implemented as switches in `tokagym/env.py` (`STRUCTURAL_DEFAULTS`), off in the visible plant and on in `tests/sealed/perturbations.json`: pressure, current-peaking and resistivity drifts, Poisson pressure crashes, supply offsets, slew limits and sticking, dead channels, flux-loop drift, non-uniform passive resistance and a beta_p-constrained profile.
- `instruction.md` names these categories without values and allows training on wider ranges.
- Fix 3 is implemented with `environment/data/fair_mast/derive_ranges.py` and `environment/data/fair_mast/calibrate_profiles.py`: the visible plant spans the interquartile range and the sealed plant the 5-95% range of MAST double-null flat-tops at 550-850 kA.
- The data moved the base plasma resistivity from 1e-6 to 3.8e-7 Ohm m (MAST's median flat-top resistance), raised the probe offsets and narrowed the resistivity range; the base case's beta_p and li_3 sit near MAST's 5th percentile.
- Fix 4 is implemented in part: 700-step episodes, a new `mastu_sequence` case (current ramp, 10 cm shift, change to the P4-scaled shape, ramp-down; 1200 steps) and a handover pre-roll under a feedforward hold.
- Shape is still measured on the psi_N = 0.995 surface rather than the last closed flux surface; the gaps, which MAST-U actually controls, are measured on the separatrix.
- A check on 8 reference and sealed states found closed contours up to psi_N = 1.0, but with point counts jumping between about 130 and 297 near the X-points; the 0.995 surface lies on average 0.4-0.7 cm inside the boundary (up to 2-4 cm at the X-points), and since targets and measurement use the same surface the offset largely cancels.
- The Super-X case and a feedback-controller handover are not done: the first needs a FreeGSNKE feasibility study, the second the PID's virtual circuits in the verifier image.
- Fix 5 is implemented as a latency gate (median per-step computation, measured against a null policy, at most 0.5 ms) and a 1,000,000-parameter cap in `contract.md`; the real-time sensor subset waits for UKAEA's list.
- The smaller corrections are applied to `references/NATURE2022.md` and the README's limitations.
- The deployable PID package and the starter were not re-run under v0.2; their v0.1 numbers stay in the README as history.

## Findings while implementing (2026-10-03)

- The privileged PID never fed back on the lower X-point height (the v0.1 baseline left it out of its virtual circuits), so it drifted by several cm; with that feedback, now the default, it holds the gaps and X-point height within 1 cm.
- On sealed seeds 1001-1002 it keeps the plasma in 12 of 13 scored episodes but succeeds in 1, because the 32-point boundary RMS stays at 1.4-2.1 cm while the gaps, X-point height and current are within tolerance.
- The verifier replays the hold-current anchor end to end (14/14 plasma lost, `reward` 0.0, `policy_compute_ms` 0.13).
- FreeGSNKE 3.1.0 drops `ny` when it copies a profile, which caused the "'ny'" failures; `tokagym.env.patch_freegsnke` fixes it.
- Profile updates double the cost of a step, so slow drifts are sent in 0.2% increments.
- The X-point height uses the candidate nearest the target, because the lowest saddle point can be far from the plasma.

## Decision: the success criterion (2026-10-03)

- Success keeps 1 cm on both the 32-point boundary RMS and the gaps, and 3% on the plasma current, as implemented.
- The 1 cm is the MAST-U team's controller requirement, "small steady-state error (about 1 cm or less)" (Lvovskiy et al. 2025), which is 2% of MAST-U's 0.5 m minor radius against the 8% of Degrave et al.'s 2 cm on TCV.
- MAST-U states it for the controlled gaps, so applying it to the whole-boundary RMS is the stricter reading, and it keeps the privileged PID at 1 success in 13 on sealed seeds 1001-1002 (12 of 13 kept).
- The alternative considered was 2 cm on the 32-point RMS (Degrave et al.'s shape aim), under which the reference would succeed in 7 of 13; switching later needs only `SUCCESS_SHAPE_M` in `tokagym/metrics.py` and the METRICS.md text.

## Main gap

Degrave et al. trained in the FGE simulator and were judged on real TCV discharges, so their train/test gap was structural: the hardware and plasma did things the simulator did not model.
This task trains and is judged in the same FreeGSNKE plant, and `tests/sealed/perturbations.json` only widens the knobs of `VISIBLE_RANDOMIZATION` in `tokagym/env.py` (1.5-2x, supply delay 0-2 steps instead of 0-1).
An agent can close most of that gap by training on wider ranges, and `instruction.md` is ambiguous about whether this is allowed.

## Keep

- The free-boundary evolutive Grad-Shafranov plant with passive structures and vertical instability, raw magnetics in and coil voltages out.
- Hidden plasma parameters, randomized per episode and never observed, as in Degrave et al.
- Privileged information at training time only (asymmetric critic).
- One controller for all cases, which is harder than Degrave et al. (one policy per experiment) and more useful in operation.

## Fixes

### 1. Scoring (first; verifier and `env.py` only)

- Define episode success as: plasma kept, steady-state shape error ≤ 1 cm, plasma-current error ≤ 3%, and no limit termination.
- The 1 cm is the MAST-U team's controller requirement: "small steady-state error (about 1 cm or less)" with "fast settling response time (about 100 ms)", while "A small transient error was not requested" (Lvovskiy et al. 2025).
- MAST-U experiments reach this level: about 2 cm steady-state error on a double-null shot (#46226) and about 1 cm on a Super-X shot (#47577) (Anand et al. 2024).
- This is stricter than Degrave et al.'s 2 cm on TCV: 1 cm is 2% of MAST-U's achieved 0.5 m minor radius, while 2 cm was 8% of TCV's 26 cm vessel half-width.
- MAST-U's real-time shape reconstruction is only required to be accurate to < 2-3 cm (Anand et al. 2024), so the simulator can check 1 cm where the real machine cannot measure it directly.
- MAST-U specifies errors per controlled quantity (inner and outer gap, X-point height), while the task averages over 32 boundary points, so apply 1 cm to both.
- Score the steady state from 100 ms after the last target change or disturbance instead of the last half of the episode, matching the settling requirement.
- With 250 ms episodes that leaves 50 ms for the vertical shift and nothing for the current ramp (it ends at 150 ms), so episodes must get longer (fix 4).
- By case averages, only the privileged PID on the P4 and pressure-drop cases (0.62 and 0.68 cm) is under 1 cm today.
- No published MAST-U plasma-current tolerance was found, so the 3% stays from Degrave et al. (5 kA of 150 kA).
- Report `reward` as the success fraction over all (case, seed) episodes next to the existing raw metrics; it is a measured quantity, so the TASK_STANDARD.md "no aggregate" exception is no longer needed.
- Today a plasma 15 cm off target still counts as kept, because loss requires a 20 cm shape RMS (`LOSS_SHAPE_RMS_M` in `tokagym/env.py`).
- Make coil-current limits terminate the episode, as Degrave et al. did: "Termination conditions are also used to enforce operational limits ... on maximum coil current or the edge safety factor."
- The deployable PID's completed episodes spend 17-18% of steps outside the limits, which machine protection would not allow.
- Separate solver failures from plasma loss: void an episode that stalls while within tolerance, count it as a loss otherwise, and withhold the reward if more than about 20% of episodes are void.
- All 7 failed sealed episodes of the privileged PID are solver stalls (README section 6 table), a noise level larger than the differences between controllers.

### 2. Structural mismatch in the sealed plant

- Add mismatches that wider training ranges cannot cover; `nlstepper` already accepts new profile parameters every step (`profiles_parameters` in `tokagym/env.py`).
- Time-varying plasma: βp and internal-inductance drift, and fast ELM-like pressure crashes.
- Power supplies: a per-episode voltage offset (modelled by Degrave et al.), slew-rate limits, and coils sticking near zero current (reported by Degrave et al.).
- Sensors: dead or biased probes and slow flux-loop integrator drift.
- Modelling error: a different profile parametrization at test time and non-uniform passive-structure resistance.
- State these categories in `instruction.md` without their values, and explicitly allow training on wider ranges than the visible ones.

### 3. Data-grounded ranges

- Degrave et al. fitted their randomization ranges to TCV data; here the ranges and the voltage/current limits are organizer-chosen (`tokagym/machine.py` marks the limits provisional).
- Use FAIR-MAST (11,573 public MAST shots with EFIT and magnetics) for βp, internal-inductance and plasma-current ranges, for real βp/li time traces as disturbances, and for sensor noise levels.
- MAST is MAST-U's predecessor with a different divertor, so these are proxies rather than MAST-U values.
- The data are downloaded to `environment/data/fair_mast/` (git-ignored `raw/`; the task's `.dockerignore` keeps the folder out of the images, since the sealed ranges derive from it); see `environment/data/fair_mast/README.md`.
- FAIR-MAST is licensed CC BY-SA 4.0, so ranges derived from it and committed to the repository need attribution and the same license.
- Ask UKAEA for a few MAST-U EFIT++ reconstructions to use as handover states and targets; FreeGSNKE has already been validated against them (Pentland et al. 2025).

### 4. Degrave-like, MAST-U-relevant targets

- All six cases derive from one example equilibrium: a 2 cm shift, a 15% current ramp, and one similarity-scaled copy (`tokagym/cases.py`).
- Add one sequence episode modelled on Degrave et al.'s capability demonstration: handover, current ramp, elongation increase (higher vertical growth rate), 10 cm vertical shift, divert with X-point control, ramp-down.
- Add a conventional-to-Super-X divertor change, MAST-U's research use of coils D1-D7, after checking that FreeGSNKE reaches Super-X shapes with this machine description.
- Start episodes from a conventional controller that hands over to the learned one, as in Degrave et al.
- Measure shape on the last closed flux surface, as Degrave et al. did, instead of the ψN = 0.995 contour; this also removes the set-point offsets in the PID baseline.
- A 1 s episode is 2,000 steps (20-100 min on the nonlinear solver), so train with the linear stepper or chunked episodes (Tracey et al. 2023) and score fewer, longer nonlinear episodes.

### 5. Real-time deployment

- Add a single-core ONNX latency gate (Degrave et al.: 50 µs) and a parameter cap.
- Restrict observations to the sensors MAST-U's real-time control system receives, if UKAEA can provide the list (Degrave et al. used a subset of TCV's sensors).

## Smaller corrections

- `environment/references/NATURE2022.md` says 1,000 parallel environments and days to weeks per task; the Nature paper reports 5,000 actors and 1-3 days, so those figures may come from Tracey et al. 2023.
- The same note lists randomized actuator delays; in Degrave et al. the delays are fixed values identified from data, and only a power-supply offset is randomized per episode.
- README section 7 says the privileged PID stalled in 6/18 sealed episodes, while the section 6 table lists 7.
- Degrave et al. found that "the main benefit comes from the recurrent design in the critic"; the starter's SAC critic is not recurrent, which may explain its value overestimation.

## Suggested order

Fix 1 first, since it removes the scoring noise for everything else; then fixes 2-4 with the task author; then fix 5.

## References

- Degrave et al., Magnetic control of tokamak plasmas through deep reinforcement learning, Nature 602, 414-419 (2022).
- Tracey et al., Towards practical reinforcement learning for tokamak magnetic control, arXiv:2307.11546 (2023).
- Lvovskiy et al., Framework for assessment of magnetic equilibrium controller performance on the MAST Upgrade spherical tokamak, Plasma Phys. Control. Fusion 67 (2025), doi:10.1088/1361-6587/ade3fd.
- Anand et al., Real-time plasma equilibrium reconstruction and shape control for the MAST Upgrade tokamak, Nucl. Fusion 64, 086051 (2024).
- Pentland et al., Validation of the static forward Grad-Shafranov equilibrium solvers in FreeGSNKE and Fiesta using EFIT++ reconstructions from MAST-U, Physica Scripta 100, 025608 (2025).
- FAIR-MAST (UKAEA), data licensed CC BY-SA 4.0: https://www.ukaea.org/service/fair-mast/

# Tokamak magnetic control: single-checkpoint AutoResearch

## 1. Overview

Develop **one learned closed-loop magnetic controller** for a MAST-U-like tokamak:
raw magnetic diagnostics in, twelve poloidal-field coil voltages out, 2 kHz, on
seven frozen cases (shape holds, current ramp, vertical shift, pressure drop, and a
Degrave-style sequence). The plant is FreeGSNKE's free-boundary evolutive
Grad-Shafranov solver with vessel eddy currents and vertical instability. Training may
use any amount of interaction with the visibly randomized plant; the independent
verifier replays a **sealed** plant with wider, FAIR-MAST-grounded ranges and mechanisms
the visible plant leaves out (time-varying plasma, supply non-idealities, sensor faults,
modelling differences, a handover). The score is the fraction of sealed episodes that
keep the plasma within the MAST-U controller requirement: steady-state shape and gaps
within 1 cm, plasma current within 3%. Robustness to plant variation the agent cannot see
is the simulation proxy for the sim-to-real gap that decides whether a controller can be
deployed on a real device (Degrave et al., Nature 2022).

v0.2 (2026-10-03) applies the verifier's review, `docs/FIX_PROPOSAL.md`; section 6 keeps
the v0.1 tables for history.

- Native Harbor task: `instruction.md`, `task.toml`, `environment/`, `tests/test.sh`.
- Simulator: FreeGSNKE 3.1.0 (LGPL-3.0), MAST-U-like description vendored from the
  pinned upstream commit; **no large data**, no downloads, no credentials.
- Environment: `environment/tokagym/` (`tokagym-mastu/v0.2`): 585-dim diagnostic
  observation, 12 coil voltages, 700 steps x 0.5 ms (1200 for the sequence), visible vs
  sealed plant, coil-current limits terminate the episode.
- Delivery: [contract](environment/contract.md), resident JSONL `infer.sh`
  (`tokamak-policy/v1`), one package for all cases, no torch in the verifier.
- Metrics: [`reward` = success fraction under the MAST-U tolerances, plus per-case shape,
  gap, X-point and current errors](environment/evaluation/METRICS.md) (`tokamak-shape-metrics/v2`);
  a latency gate of 0.5 ms median computation per step.
- References: privileged PID/virtual-circuit controller from FreeGSNKE's pulse
  design tool (`environment/baseline/pid/pid_reference.py`), its deployable
  estimator-based package (`baseline/pid/build_pid_package.py`), hold-current and
  random-init anchors, BC/residual SAC starter with asymmetric critic and TD3+BC
  normalization (`environment/baseline/starter/`).

The workspace follows [TASK_STANDARD.md](../TASK_STANDARD.md).

## 2. Data

No large dataset in the images: the MAST-U-like machine description (0.7 MB) is vendored from FreeGSNKE under
`environment/data/machine_configs` (LGPL-3.0) and copied into the images; see
[environment/data/README.md](environment/data/README.md). No `TASK_DATA_DIR` or `TASK_PRIVATE_DIR` is needed; the
sealed perturbation family lives in the verifier image (`tests/sealed/`).

The randomization ranges are grounded in FAIR-MAST (UKAEA's open MAST archive, CC BY-SA 4.0) by the
developer-side tools in [`environment/data/fair_mast/`](environment/data/fair_mast/README.md): `download_dataset.sh` fetches the summary table, EFIT
traces and loop voltages of 11,052 MAST plasma shots (3 GB, git-ignored; the task's `.dockerignore` keeps the folder out of every image, since the sealed ranges derive from it) and
`derive_ranges.py` writes `derived_ranges.json`, which `tokagym/env.py` and `tests/sealed/perturbations.json` cite.

## 3. Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md`, `/workspace/contract.md` | the prompt; the controller package and `infer.sh` protocol |
| `/workspace/tokagym/` | the simulator environment, the seven cases, machine wrapper, metrics, the switches of the sealed-plant mechanisms |
| `/workspace/data/` | the vendored machine description |
| `/workspace/baseline/starter/` | BC-from-PID + SAC / residual SAC; NumPy-deployable actor, atomic publish (v0.1: 15/30 sealed episodes kept; not re-run under v0.2) |
| `/workspace/baseline/pid/`, `/workspace/baseline/hold_current/` | privileged PID, deployable PID package, no-control anchor (v0.2 results in section 6) |
| `/workspace/references/` | notes on the Nature 2022 recipe and the deployable PID |
| `/workspace/evaluation/` | `METRICS.md`, development replay (`harbor_verify.py`), learning curves |
| `/workspace/submission/`, `/workspace/research/` | the current package, and writable scratch on a large disk (`TASK_RESEARCH_DIR`) |

## 4. Evaluation

`tests/test.sh` replays the published package on the seven cases with five sealed seeds and the sealed plant
(`tests/sealed/perturbations.json`, `sealed-v0.2`), with no network, one supervised process per episode. It writes
`reward`, the fraction of non-void episodes that keep the plasma within the MAST-U steady-state tolerances (shape and
gaps 1 cm, plasma current 3%), next to `policy_compute_ms` and per-case success/feasible fractions and steady-state
shape, gap, X-point and current errors in `/logs/verifier/reward.json`. The success fraction is a measured quantity, so
the task no longer needs the TASK_STANDARD.md exception. An invalid package (policy error, or a median per-step
computation above 0.5 ms) scores 0; more than 20% void episodes (numerical stalls of a plasma within tolerance)
withhold the score. A replay is about 27,000 steps; budget 2-5 h at eight workers.

## 5. Running

```bash
export TASK_RESEARCH_DIR=/large-disk/runs/tokamak-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p tokamak-magnetic-control -a <agent> -m <model> --allow-agent-host <model API host>
```

The external budget is set by the controller: a 48 h agent budget in `task.toml` (the standard default), one GPU and 64 CPU cores for
development, a 10-hour verifier with 32 CPU cores. Development commands (maintainers):

Harbor builds the development environment from `environment/docker-compose.yaml`
(target `development`) and the separate verifier from `tests/docker-compose.yaml`
(target `verifier`, which alone carries `tests/sealed`); the same two targets by
hand:

```bash
docker build --target development -t tokamak-dev -f environment/Dockerfile .
docker build --target verifier    -t tokamak-verifier -f environment/Dockerfile .

# environment smoke (inside tokamak-dev, PYTHONPATH=/workspace)
python /workspace/evaluation/smoke_env.py --cases mastu_diverted_hold --steps 20
# random-init package -> development replay
python /workspace/baseline/starter/init_random.py --actor-dir /workspace/research/r0 --submission /workspace/submission
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/check-001 --cases mastu_diverted_hold --seeds 0 --workers 1
# starter training (32 simulator workers, one GPU or CPU learner)
python /workspace/baseline/starter/train.py --run /workspace/research/run-001 --workers 32 --hours 6
```

Trusted verification: `tests/test.sh` inside `tokamak-verifier` with the package
mounted at `/workspace/submission` and `/logs/verifier` writable (see
[evaluation README](environment/evaluation/README.md)).

CPU-bound: one nonlinear evolutive step takes 0.6-3 s on 2-4 threads on the maintainers' shared host (0.1-0.4 s on
an otherwise idle 8-core laptop; profile updates of the time-varying sealed plasma add up to 2x; the linear
re-linearized plant is only 1.3-2x cheaper), so a 700-step episode takes 5-35 minutes and 48 two-thread workers give
~2 episodes/minute. The learner needs at most one GPU (the actor is a 3x256 MLP). The verifier replays
7 cases x 5 sealed seeds (35 episodes, about 27,000 steps) with eight four-thread workers and needs no GPU;
`task.toml` gives it 32 CPU cores.

## 6. Results and validation

### v0.2 (sealed-v0.2, `tokamak-shape-metrics/v2`)

| Item | State |
| --- | --- |
| Environment, cases, metrics, verifier, contract | implemented; 19 unit tests pass; all seven cases build and 21 sealed resets succeed without fallback (local FreeGSNKE 3.1.0, no Docker) |
| Privileged PID reference | see the v0.2 table below |
| Hold-current anchor through the verifier | see the v0.2 table below |
| Deployable PID package, RL starter | not re-run under v0.2 (their v0.1 numbers below are for the old metrics and plant) |
| Formal agent session | not run |

Privileged PID with X-point-height feedback, sealed-v0.2 seeds 1001-1002 (local run; details in
`environment/baseline/pid/RESULTS.md`):

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

Hold-current anchor through `harbor_verify.py` on the same seeds: 14/14 `plasma_lost` after 29-117
steps, `reward` 0.0, `policy_compute_ms` 0.13. Five seeds (`tests/test.sh`) were not run locally: one
run of the 14 PID episodes took 1.5 h with four workers here, so a five-seed replay is several hours on this machine.

### v0.1 history (sealed-v0.1, `tokamak-shape-metrics/v1`, 500-step episodes)

The tables below are the task author's v0.1 runs. They use the old feasibility and last-half metrics, the
±10 kA solenoid limit as a measured (not terminating) limit and the old plant, so they are not comparable with v0.2.

#### Sealed table, no-training references (seeds 1001-1003, nonlinear plant, last-half window)

Privileged PID / virtual circuits (simulator-side descriptors; not a submission):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] | Notes |
| --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 2/3 | 1.66 | 2.39 | 0.11 | seed 1002: solver stall |
| mastu_high_ip_hold | 2/3 | 1.66 | 2.39 | 0.11 | identical to the base case: it is a similarity-scaled copy (same boundary, currents x1.21) |
| mastu_ip_ramp | 0/3 | - | - | - | solver stall in every seed during/after the current ramp |
| mastu_p4_shape_hold | 3/3 | 0.62 | 1.49 | 1.04 | |
| mastu_pressure_drop | 2/3 | 0.68 | 0.88 | 0.13 | seed 1002: solver stall |
| mastu_vertical_shift | 2/3 | 1.38 | 4.82 | 0.12 | seed 1002: solver stall |

"Solver stall": FreeGSNKE's nonlinear stepper stopped converging after a few hundred
steps (the episode exceeded 60 min and was killed; re-runs at low host load stalled
again). The first 80 steps of the seed-1002 base draw and 220 steps of the ip_ramp
draw are unremarkable (shape error 0.9-5 cm), so this is a late-episode solver
limitation interacting with the PID's slow drift, not a plant that cannot be
simulated. By the task rules a stall is an infeasible episode for whoever is evaluated.

Anchors (protocol submissions): hold-current (`V = R I`) loses the plasma in every
episode (18/18; plasma lost after 48-362 steps, i.e. 24-181 ms; the seed-1001 draws
survive longest). Random-init loses it in every episode after 15-52 steps. Both score
feasible fraction 0 on all cases.

Deployable PID package (estimator + PID/virtual circuits from the observation only;
v5 = per-shape virtual circuits, set-point ramp, controller rebuilt per episode):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] | Notes |
| --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 2/3 | 4.76 | 4.43 | 0.68 | seed 1003: solver stall |
| mastu_high_ip_hold | 0/3 | - | - | - | solver stall in every seed |
| mastu_ip_ramp | 0/3 | - | - | - | two stalls; seed 1002 hit the 180 s step budget at step 249 |
| mastu_p4_shape_hold | 0/3 | - | - | - | two stalls; seed 1003 plasma lost at step 82 |
| mastu_pressure_drop | 2/3 | 3.53 | 2.53 | 0.76 | seed 1003: solver stall |
| mastu_vertical_shift | 2/3 | 4.42 | 9.79 | 0.72 | seed 1003: solver stall |

Completed episodes sit at 2.5-5.7 cm shape error, i.e. 2-4x the privileged
controller, with 17-18% of steps outside the provisional coil-current limits. Eleven
of the eighteen episodes were killed after 15 minutes without a completed step: with
this controller the nonlinear solver stalls far more often than with the privileged
one (6/18), consistent with the vertical-loop drift described below driving the
plasma into states the Newton-Krylov stepper cannot follow. The earlier v3 package
(base-shape virtual circuits, no ramp) lost every P4 episode outright.

What the diagnostics showed (visible draws, `runs/tools/diag_*.py` traces):

- The observation targets are the frozen nominal boundary while every reset re-solves
  the plasma with perturbed profiles, so the plasma starts 1-3 cm off the targets and
  the controller must move it. With the base-shape virtual circuits applied to the P4
  shape the package lost the plasma on every P4 episode; per-shape circuits selected
  by the observation's targets fixed the shape loop (it then tracks the privileged
  controller within ~0.5 cm on the base and P4 draws).
- The vertical loop is the remaining weakness of the deployable controller: with the
  estimated current centroid (0.4-0.6 cm noise per step) the FPDT derivative term
  saturates P6, and every filter/derivative-gain variant we tried (EMA 0.12-1.0,
  k_deriv 0-5e-7) lost the X-point within 35-95 steps on the harder draws where the
  privileged controller (exact centroid and velocity) keeps P6 at 0 V. A model-based
  vertical observer is the obvious next step; it is also exactly the kind of
  improvement an RL residual or a learned estimator could deliver.

#### RL starter

Runs 005-011 (BC-from-PID + SAC, then residual SAC on the deployable PID core; linear
re-linearized plant) all collapsed once actor updates started: training-window
feasibility fell from 1.0 to 0.0 within ~4k updates and the published package was
worse than its initialization. Cause: the critic's extrapolation error outside the
demonstrated actions dominated the BC/residual regularizer. `starter/train.py` now
scales the Q term by `q_weight / mean|Q|` (TD3+BC), freezes the entropy coefficient
(`--alpha-lr 0`), publishes a SAC snapshot only if its training window beats the
demonstrator's, and exports snapshots for `evaluation/curve.py`. Run 012 (residual on
the v3 package, nonlinear plant) was stopped because the base controller itself lost
most visible-draw episodes. Run 015 (direct mode: 32 privileged demonstrations, 6000 BC steps, then SAC with the
fixed objective; nonlinear plant, 36 workers) is the run behind the sealed learning
curve below (`evaluation/curve.py`, seeds 1001-1003, every second snapshot; each
point is an independent replay of the exported snapshot through the verifier).

| Snapshot (episodes / env steps / SAC updates) | Mean feasible | Mean shape RMS [cm] | Per case: feasible / shape cm |
| --- | --- | --- | --- |
| ep 500 / 43k / 91 (= behaviour-cloned actor) | 0.83 (15/18) | 2.19 | base 1.00/2.36, P4 0.33/2.90, high-Ip 1.00/1.95, ip_ramp 0.67/0.98, vertical 1.00/2.82, pressure 1.00/2.11 |
| ep 600 / 71k / ~14k (SAC actor updates on) | 0.00 (0/18) | - | plasma lost after 40-89 steps in 15 episodes, 3 solver stalls |

The behaviour-cloned actor alone (a 3x256 MLP on the 585 diagnostics, no simulator
access, ~50 us per step) already beats the deployable PID on the sealed set (15/18
feasible at 2.2 cm vs 6/18 at 2.5-5.7 cm) and completes the ip_ramp episodes on
which both PID controllers stalled the solver. Its weak spot is the plasma current:
7.5% error on the 750 kA case and 12.4% during the current ramp (the PID references
hold 0.1-1%), so the shape is held while Ip drifts. The three lost episodes are two
solver stalls and one FreeGSNKE `ConstrainPaxisIp ... 'ny'` exception (a simulator
bug that counts as infeasible). SAC did not improve on it: once actor updates
started (after 3000 critic-only updates) the training-window feasibility fell from
5/10 to 1/10 within ~10k updates while Q(s, pi(s)) climbed from 20 toward the clip
at 100 and the critic loss grew, the same over-estimation spiral as in runs 005-011
despite the normalized objective. The publish-if-better rule kept the BC actor as
the published package. The published BC package was also run through the trusted verifier entry
(`tests/test.sh`, all five sealed seeds, verifier image):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] | Seeds 1004 / 1005 |
| --- | --- | --- | --- | --- | --- |
| mastu_diverted_hold | 3/5 | 2.36 | 3.18 | 0.93 | solver stall / plasma lost at step 47 |
| mastu_high_ip_hold | 3/5 | 1.95 | 3.17 | 7.48 | solver stall / plasma lost at step 55 |
| mastu_ip_ramp | 2/5 | 0.98 | 1.35 | 12.43 | solver stall / plasma lost at step 47 (1002 also stalled) |
| mastu_p4_shape_hold | 1/5 | 2.90 | 4.30 | 5.11 | solver stall / solver stall (1001 stalled, 1003 `ny` exception) |
| mastu_pressure_drop | 3/5 | 2.11 | 2.47 | 1.14 | solver stall / plasma lost at step 47 |
| mastu_vertical_shift | 3/5 | 2.82 | 4.55 | 1.12 | solver stall / plasma lost at step 47 |

Seeds 1001-1003 reproduce the curve point exactly (the replay is deterministic). Seed
1004 stalls the solver in all six cases and seed 1005 loses the plasma within 25 ms in
five of them; both are low-pressure draws (axis pressure about 60% of nominal, seed
1004 also with +20% passive resistance and a one-step supply delay), a regime the 32
demonstrations barely covered. Three seeds are a thin estimate; the verifier's five
are the number that counts.

The ep 600 row above records the degradation on the sealed seeds: the
SAC-updated actor loses the plasma within 20-45 ms in every episode. The run ended at
its 5 h budget (78k environment steps, 17.7k updates) without recovering. Conclusion for v0.1: the starter delivers a deployable imitation policy; the
RL improvement step is open, and the evidence points at off-policy critic
extrapolation with too little on-policy data (~3 completed episodes per minute on
this host) rather than at the environment. Candidate fixes for v0.2: on-policy
fine-tuning from the BC actor (PPO/MPO-style, as in the Nature recipe), a much larger
BC weight or DAgger-style relabelling with the privileged PID, or conservative
offline RL on the demonstration replay.

Numbers above are from the maintainers' runs on a shared 208-core host; they are
reference points, not leaderboard entries.

Layout and integration checks: [`docs/VALIDATION.md`](docs/VALIDATION.md).

## 7. Limitations

- "MAST-U-like" geometry and provisional voltage/current limits except the solenoid
  (MAST-U's ±55 kA, Fishpool et al. 2013); no engineering validation of the others.
  The public FreeGSNKE description is the only open free-boundary machine model with
  probes; SPARC/ITER examples lack diagnostics.
- The randomization ranges come from MAST (FAIR-MAST), MAST-U's predecessor with a
  different divertor; MAST-U data are not public. The base case's beta_p (0.19) and
  li_3 (0.71) sit near the 5th percentile of MAST double-null flat-tops, so the ranges are
  relative spreads around a low-beta (L-mode-like) case; a high-beta H-mode case would be a
  natural addition. Sensor statistics come from EFIT fit residuals and bound the noise from
  above. Supply non-idealities, flux-loop drift, passive-filament spread and the profile
  constraint switch are organizer choices with no data behind them.
- FreeGSNKE has no current diffusion or transport; profile parameters are
  prescribed and randomized rather than evolved.
- The FPDT Lao85 double-null equilibrium is excluded (its nonlinear evolutive step
  did not converge in acceptable time), and a limited-plasma case is deferred (the
  upstream limited-current set could not be reached by continuation from the base
  equilibrium). The 750 kA case is a similarity-scaled copy of the base shape; the
  P4-scaled (x1.2) case is the only distinct diverted shape in v0.1.
- FreeGSNKE's nonlinear stepper occasionally stops converging late in an episode
  (a "solver stall") and raises a `ConstrainPaxisIp ... 'ny'` AttributeError on rare
  steps. In v0.1 all 7 of the privileged PID's failed sealed episodes were stalls (the
  v0.1 table; an earlier version of this note said 6/18). v0.2 voids a stall of a plasma
  within tolerance and withholds the score above 20% void episodes; the stepper itself is
  unchanged.
- The handover is a feedforward hold (V = R I) for up to 20 steps, not a feedback
  controller handing over as on TCV; the PID lives in `baseline/` and needs virtual
  circuits, which the verifier image does not carry.
- No Super-X case yet: whether FreeGSNKE reaches Super-X shapes with this machine
  description is unverified. The real-time sensor subset of MAST-U's control system is
  not public, so the observation keeps every probe.
- The 750 kA case is an exact similarity copy of the base shape, so a controller
  that scales with the currents (the PID references) gets identical shape numbers on
  both; it still tests a learned policy's sensitivity to signal amplitude.
- The latency gate measures the policy's computation through the JSON protocol (round
  trip minus a null policy's), so it resolves about 0.1 ms, not Degrave et al.'s 50 µs;
  the parameter cap is checked by review.
- Sealed perturbations are in `tests/sealed/`, which is copied only into the
  verifier image; they are hidden from the agent's container, not from Git.

## 8. Citation and licenses

- Degrave et al., Magnetic control of tokamak plasmas through deep reinforcement learning, Nature 602 (2022).
- Tracey et al., Towards practical reinforcement learning for tokamak magnetic control (2023).
- FreeGSNKE (Amorisco et al., Physics of Plasmas 2024), LGPL-3.0; machine description from its repository.
- Lvovskiy et al., Framework for assessment of magnetic equilibrium controller performance on the MAST Upgrade spherical tokamak, PPCF 67 (2025): the 1 cm / 100 ms requirement.
- Anand et al., Real-time plasma equilibrium reconstruction and shape control for the MAST Upgrade tokamak, Nucl. Fusion 64, 086051 (2024).
- Fishpool et al., MAST-upgrade divertor facility and assessing performance of long-legged divertors, J. Nucl. Mater. 438 (2013): the ±55 kA solenoid range.
- FAIR-MAST (UKAEA), data CC BY-SA 4.0; `environment/data/fair_mast/derived_ranges.json` is derived from it under the same licence.

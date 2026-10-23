# Continuously improve one tokamak magnetic controller

## Task and goal

Research **one closed-loop magnetic controller** for the MAST-U-like tokamak simulated by
FreeGSNKE. The controller receives raw magnetic diagnostics and target descriptors, and commands
the twelve poloidal-field coil voltages at 2 kHz. Improve it across the complete, fixed case suite
in `/workspace/tokagym/cases.py`, using one jointly trained parameter set for every case. There are
seven cases:

- three shape holds;
- a plasma-current ramp;
- a vertical shift;
- an unannounced pressure drop;
- a sequence modelled on Degrave et al. (Nature 2022): current ramp, 10 cm vertical shift,
  change to a second shape with X-point motion, ramp-down.

Every target is an equilibrium this machine can hold: a vertical move targets the plasma held higher
by the vertical coil P6 (the shaping coils are up-down symmetric pairs, so it is not a rigid shift).

The score is the fraction of episodes that succeed: the plasma is kept for the whole horizon
without hitting a coil-current limit, and after a 100 ms settling time the boundary shape, the
inner and outer gaps and the X-point height stay within 1 cm and the plasma current within 3%.
These are the MAST-U team's controller requirements. `/workspace/evaluation/METRICS.md` defines
them exactly and lists the per-case physical metrics reported alongside. Native rewards are
training signals, not benchmark scores. Aim to outperform the classical `shape_vc`
reference on every case, under the randomized plant, not only in the nominal simulator.

| controller | sealed successes | sealed episodes kept |
| --- | --- | --- |
| hold-current (`V = R I`) | 0/105 | 0/105 |
| deployable full-shape virtual-circuit controller (`baseline/shape_vc/`) | 48/105 | 79/105 |

All on the 105 sealed episodes (seven cases, fifteen seeds) of the current suite (`tokagym-mastu/v0.3`,
`tokamak-shape-metrics/v4`). `shape_vc` is the classical reference to beat:
a linear magnetic reconstruction, least-squares virtual circuits on the whole boundary and a
delay-robust vertical loop, NumPy only. The starter trains a residual SAC policy on top of it
(5 h, 311k steps: 16/34 sealed under the earlier 35-episode metric v3, the pipeline verified but no gain over the base
yet).

## Workspace

- `/workspace/contract.md`: the controller package, `infer.sh` protocol and inference limits.
- `/workspace/tokagym/`: the simulator environment, cases, machine wrapper and metrics
  (`tokagym/env.py` holds the visible randomization).
- `/workspace/data/`: the vendored MAST-U-like machine description.
- `/workspace/baseline/`: verified implementations with their sealed results:
  - `starter/` is residual SAC on a deployable base controller (`shape_vc` by default), or behaviour
    cloning from the privileged PID plus SAC;
  - `shape_vc/` is the deployable full-shape virtual-circuit controller (classical reference,
    with its estimator data pipeline and built package);
  - `pid/` holds the privileged PID and the deployable PID package;
  - `hold_current/` is the no-control anchor.
- `/workspace/references/`: notes on the Nature 2022 recipe and on the deployable PID.
- `/workspace/evaluation/`: `METRICS.md`, development replay (`harbor_verify.py`) and learning
  curves.
- `/workspace/research/`: writable space for runs, demonstrations, checkpoints and notes.

## Rules

- This is a **known-task, randomized-plant control problem**. You may interact with every
  case configuration as often as you like, with the visible randomization in
  `/workspace/tokagym/env.py`:
  - profile parameters and plasma resistivity;
  - passive structure resistance and initial coil currents;
  - diagnostic offsets and noise;
  - supply delay.
- The independent verifier replays the same cases with a **sealed** plant: wider ranges than the
  visible ones (grounded in MAST data), unseen seeds, and mechanisms the visible plant leaves out.
  Their categories are known, their values are not:
  - time-varying plasma: drifts of pressure, current-profile peaking and resistivity, and fast
    pressure crashes (ELM-like);
  - power supplies: per-episode voltage offsets, slew-rate limits, and supplies that stick at 0 V
    near zero coil current;
  - sensors: dead magnetic channels that read 0, and flux-loop integrator drift;
  - modelling differences: a beta_p- instead of a pressure-constrained profile, and non-uniform
    passive-structure resistance;
  - a handover: the plasma first evolves for a few steps under a feedforward hold.

  Robustness to plant variation you cannot see is therefore the scored quantity. It is the
  simulation proxy for the sim-to-real gap. You may train on any randomization, including wider
  ranges and these mechanisms: their switches are in `/workspace/tokagym/env.py`.
- All case definitions, reference results and evaluator code are visible. Run development
  evaluation directly and use it for analysis and model selection.
- During a scored episode, weights and normalization are fixed. Recurrent state, integrators and
  non-learning planning are allowed and reset each episode. Do not turn the scoring replay into an
  adaptation run.
- Preserve the evaluator's machine description, grid, time step, horizons, observation definition,
  actuator and coil-current limits (an episode ends when a coil current leaves its limit) and
  metrics. You may change:
  - architecture, representation and optimization;
  - reward shaping, experience collection (including the linear solver mode for cheap rollouts)
    and curricula;
  - surrogate or world models;
  - inference computation.

  The nonlinear solver is the ground truth for scoring. `/workspace/tokagym` is your training copy:
  you may modify, accelerate or replace it. The verifier scores with its own copy and the nonlinear
  solver.
- Submit one controller serving all seven cases. Task-conditioned components and jointly trained
  input-dependent computation are allowed; independent per-case controllers behind a dispatcher
  are not.
- There is no network.

## Submission

Keep one logical checkpoint, one controller's parameters serving all seven cases, at
`/workspace/submission`. The deployed controller must run without PyTorch (NumPy or ONNX). Keep its
inference cheap, as on a real plasma control system: `/workspace/contract.md` sets a 0.5 ms median
per-step computation (a slower package scores 0) and at most 1,000,000 learned parameters.

The package may be read at any moment, so replace it atomically, as the starter's `publish.py`
does. Keep experiment archives, optimizer state and old models under `/workspace/research`,
outside the published package.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

`baseline/shape_vc/RESULTS.md` already has the reference on the full development suite (visible
randomization, seeds 0–2, per episode): compare candidates with it instead of replaying the baseline.
Start by checking your own pipeline cheaply: one real training update, a publication, and a one-case,
one-seed replay (`--cases mastu_diverted_hold --seeds 0`). A full development replay runs its 21 episodes
in parallel by default and takes about an hour. Then repeatedly:

1. analyze per-case results and feasibility rates;
2. form a testable hypothesis;
3. implement and train a candidate;
4. evaluate the same checkpoint on all cases with several seeds;
5. publish the strongest complete, replayable candidate;
6. select the next experiment.

This is continuing research, not a finite deliverable. Do not stop after a training run, a
successful publication or a score target. Maintain concise research notes with:

- per-case metrics and feasibility;
- the hypothesis;
- training provenance and simulator step counts;
- wall-clock costs.

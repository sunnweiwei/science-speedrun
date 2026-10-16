# Manual reproduction status — 2026-09-06

This is an interim experimental record, **not a successful reproduction claim**.
No AutoResearch agent has been launched. Run outputs are on the host data disk
under `/mnt/workspace/hydrogym/reproduction/`.

Latest measured run status is in [GOAL_PROGRESS.md](GOAL_PROGRESS.md); the
timestamped running-status sections below are historical. The first nine-run
batch used the earlier terminal-at-horizon behavior. Corrected native runs now
bootstrap the critic at the external time limit, matching the public author's
StepCounter semantics; see [TIME_LIMIT_AUDIT.md](TIME_LIMIT_AUDIT.md). Their
results must not be pooled with that earlier batch as independent seeds of one
configuration. The trained JAX5000 PPO's same-GPU replay returned27.94386697
with zero actions on the evaluated trajectory; that is not evidence of enhanced
mixing or of the paper's stronger GPPO result.

## Completed first full-budget batch: performance target not met

All nine runs reached 100,000 native control steps / 500 training episodes.
The numbers below use each seed's **final**, not selected-best checkpoint.
All nine training-to-independent-final/best evaluation workflows completed.
Their 207 archived, hashed files are in
`data/baseline-results/public-pinball-20260905` and explicitly labelled
non-accepted candidate results.

| Algorithm | Seeds | Native final return, mean ± sample SD | Full-window drag reduction, mean ± sample SD |
| --- | --- | ---: | ---: |
| PPO | 42, 43, 44 | -764.66 ± 314.78 | 10.14% ± 42.08 percentage points |
| DDPG | 42, 43, 44 | -396.94 ± 105.64 | 62.61% ± 18.79 percentage points |
| TD3 | 42, 43, 44 | -506.66 ± 271.00 | 50.12% ± 38.17 percentage points |

These do **not** reproduce the paper's approximately 90% Pinball drag reduction
or its consistent three-seed convergence. The selected-best TD3 seed 44 reduced
full-window drag by 85.82% and last-half drag by 91.19%, but has large lift
fluctuations; one chosen seed/window is not sufficient evidence of reproduction.
Six new controlled recipe ablations are running; see RECIPES_20260906.md.

Full public-asset materialization is complete: all 89 directories / 1,469 files
passed pinned Git/LFS checks. Task CPU tests pass in the development runtime;
native shared-policy testing and the full scientific release remain outstanding.

The development image also completed its own 1,200-step PPO training and fresh
final/best replays. A separately exported TD3 actor was relocated and independently
served over the production JSON-lines interface for 200 native CFD actions:
return -198.4936591981 versus the original -198.4936679575 (difference 8.76e-6).
This validates the delivery/evaluation path, not the paper-performance target.

Cylinder Re3900's complete zero-control check (600 CFD substeps/action, 200
actions) returned -566.9162190, with mean CD=1.62350477 and RMS CL=1.34621355.
The separate two-case shared-checkpoint train/resume/replay check completed:
one shared model received 400 training controls per case, was relocated, and
completed two independent 200-action replays without package mutation. This
verifies the shared-policy starter contract, not multi-case competitive control.

Grid headers additionally confirm exact paper-table cell counts for 2D Pinball
(340,422), 3D Pinball Re150 (16,465,722), 2D Cylinder Re3900 (934,732), and both
available 2D Cavity grids (213,940). Some domain/refinement descriptions in the
supplement differ from these public assets. Cell-count agreement is supporting
evidence, not proof that every physical or control parameter is identical.

## Current public-source reconstruction (supersedes the historical status below)

The user authorized performance-first reconstruction from public sources without
contacting authors. See [PUBLIC_REPRODUCTION.md](PUBLIC_REPRODUCTION.md) for all
effective settings and differences. Source/data hashes have been reverified.
CPU tests verify PPO/DDPG/TD3 actual parameter updates, DDPG BN running statistics,
checkpoint reload/deterministic inference, and rejection of incompatible physics.
The task's 21 structural tests passed with one dependency-related skip.

Native 1,200-step checks: PPO and TD3 completed at the old 0.5 control limit;
DDPG's post-update feedback trajectory diverged and correctly stopped. A separate
PPO evaluation container reproduced its step-400 evaluation return and drag/lift
exactly. The new automatic training-to-fresh-process-evaluation workflow also
completed successfully in `hg-public-pinball2d-workflow-check-20260905/`.

Fixed-control checks, all complete 200-step native episodes:

| Physical controls | Raw return | Full mean drag | Last-half mean drag |
| --- | ---: | ---: | ---: |
| `[0,0,0]` | -587.88099 | 2.905470 | 2.905670 |
| `[0,-0.25,0.25]` | -72.43182 | 0.332960 | 0.248932 |
| `[0,-0.5,0.5]` | -73.05342 | -0.120065 | -0.248648 |
| `[-0.25,-0.25,-0.25]` | -2869.96976 | 3.204085 | 3.036739 |
| `[-0.5,-0.5,-0.5]` | -6056.05603 | 2.393484 | 0.291690 |

The 0.25 counter-rotation's reduction is 88.54% over the full window, 91.43% over
the fixed last-half window. This supports the physical attainability of the
paper's approximately 90% claim, **not successful RL training reproduction**.
Negative signed drag at 0.5 must not be presented as an artificially better score:
the native reward penalizes absolute drag and lift. The all-negative policies
have large lift penalties despite any apparent signed-drag reduction.

Main 500-episode, three-training-seed runs use the current public package's
0.25 actuator limit. Their evolving run records are separate from the old pilots.
No paper RL-performance match or full-suite validation has yet been claimed.
DDPG's 0.25 short check has also completed, including independent final/best
evaluations with exact agreement of physical metrics. Eight full-budget runs are
launched and the ninth is queued; see [RUNS_20260905.md](RUNS_20260905.md).

## Historical pilot record

**Later update, 2026-09-05:** all three 40,000-step pilots and the full 3D
zero-control episode have completed. Final checkpoint evaluations are in
`hg-pinball2d-{ppo,ddpg,td3}-40k-eval-20260905/`. These do **not** match a verified
paper protocol. The earlier running-status section below is historical. See
[STRICT_REPRODUCTION.md](STRICT_REPRODUCTION.md) for final numbers, the newly
located author code, and the unresolved exact-reproduction requirements.

## Completed: native solver and checkpoint evaluation

The official pinned MAIA container ran a complete, unmodified public-asset
Pinball_2D_Re100 zero-control episode. Measured mean drag was
**2.9052879687**, versus **2.904** in supplementary Table 3 (about 0.044% apart).
This checks the uncontrolled flow, not a learned controller or statistical
equivalence of the paper's protocol. Output: `pinball2d-zero-native/`.

After applying the paper's documented omega=1, 170 CFD steps/action and 200
actions/episode, the zero-control return was **-587.8809887800**, with mean drag
**2.9054702151**. Three resets produced identical trajectories. These are
deterministic repetitions, **not three independent training seeds**. Output:
`hg-pinball2d-zero-paper-20260905/`.

Early checkpoints from three real TorchRL learners were copied to stable paths
and evaluated in separate offline containers. All three completed a full native
CFD episode and exited successfully; no training occurred in those evaluations.

| Controller | Training control steps | Evaluation return ↑ | Mean drag ↓ | RMS lift ↓ |
| --- | ---: | ---: | ---: | ---: |
| Zero control, paper-reconstruction protocol | — | -587.8810 | 2.90547 | 0.03419 |
| PPO, seed 0 | 1,400 | -585.3787 | 2.29227 | 0.76513 |
| DDPG, seed 0 | 1,200 | -1076.2393 | 3.88413 | 1.94757 |
| TD3, seed 0 | 1,400 | -1083.4814 | 3.63455 | 2.23139 |

These are single early policy evaluations, not converged results, selected-best
scores, or confidence intervals. PPO reduced drag but increased lift variation;
its penalized return remained near zero control. The other two early policies
were worse than zero control. Do not infer paper-level success from drag alone.

Raw records: `hg-pinball2d-{ppo,ddpg,td3}-early-eval-20260905/`, including
`episodes.jsonl` and `evaluation.json`. The latter contains the exact checkpoint
training step. Stable evaluated checkpoints are `eval-snapshots/{algo}-early.pt`.

## Running: first-seed training pilots

As of 10:50 UTC, all three runs had reached 2,600–3,200 control steps with finite
losses, actual optimizer updates and published checkpoints. Each targets 40,000
control steps (200 episodes); this is a first pilot budget, not paper convergence.
An additional checkpoint audit rebuilt each seed's initial actor and confirmed
nonzero weight changes for all three algorithms (L2 changes 3.631, 24.478 and
19.051 at PPO/DDPG/TD3 steps 3,600/3,200/3,800 respectively). An update counter
alone was not used as proof that the submitted actor learned.

| Docker run / output directory | GPU | Algorithm |
| --- | ---: | --- |
| `hg-pinball2d-ppo-seed0-20260905` | 0 | PPO |
| `hg-pinball2d-ddpg-seed0-20260905` | 1 | DDPG |
| `hg-pinball2d-td3-seed0-20260905` | 2 | TD3 |

These first three runs have a `source-at-launch/` copy; later launcher invocations
freeze and mount `source/` before starting. Each also records its exact launch
command. Check live logs rather than interpreting this timestamped table as
current status.

The 3D Pinball pinned grid has 16,465,722 cells (14,578,122 leaf cells), from its
NetCDF header. `hg-pinball3d-smoke-20260905` on GPU 6 **completed successfully**:
five control steps with finite observations/rewards, 281.8 seconds including
solver initialization, container exit code 0. It is not a scored episode or 3D
learner reproduction. At 10:57 UTC, `hg-pinball3d-zero-paper-20260905` was launched
on GPU 6 for one complete 200-action uncontrolled episode under the documented
paper settings. No result is available yet. The short run measured roughly
0.026 seconds per CFD substep after initialization: extrapolating 40,000 control
steps at 200 substeps each would take roughly 58 GPU-hours, before resets and
learner overhead. This is a feasibility estimate, not a measured training cost.

## Still required before claiming reproduction

- Original per-case TorchRL training configurations and exact observation probe
  locations, or clearly justified and agreed reconstruction tolerances.
- Mesh equivalence and consistent physical measurement windows, not merely
  matching an uncontrolled drag coefficient over one episode.
- Training to the paper's learning-curve budget, independent training seeds
  (the paper uses three for these RL comparisons), and frozen-policy evaluations.
- Remaining paper comparison environments: 3D Pinball, cylinder, cavity and
  gusting airfoil, with their own protocol and resource validation. This pilot
  does not stand in for the full benchmark.
- GPPO source or the missing exact gradient-weighting/optimizer configuration.
  The checked paper archive supplies JAX PPO but no GPPO training implementation;
  a guessed GPPO would not be a faithful reproduction of Fig. 3b.

The full Harbor evaluation suite remains unvalidated. Do not convert these pilot
returns into task release scores or treat one-case checkpoints as the eventual
single-checkpoint multi-environment submission.

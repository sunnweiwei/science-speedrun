# Public-source, performance-first reproduction

The current instruction is to reproduce the paper's performance, keep its setup
as close as public information allows, and **not contact the authors**. This is
manual baseline work; no AutoResearch agent is launched. Missing unpublished
configuration is documented, not used as a reason to stop experimental work.

**Scope update, 2026-09-06:** the current requirement is full-suite coverage with
at least one competitive paper method per test case, not reproduction of the
entire PPO/DDPG/TD3 algorithm grid. The sections below describe existing Pinball
experiments; they do not define the final suite or its acceptance protocol.
See [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md).

## First comparison: 2D Pinball Re100

Target: Extended Data Fig. 2a of the [HydroGym paper](https://www.nature.com/articles/s41586-026-10917-6).
PPO, DDPG and TD3 should each learn effective drag/lift control within the plotted
500-episode window. The paper describes approximately 90% drag reduction.
This is the first case, **not the full benchmark or a substitute for 3D results**.

Sources, kept untouched on the data disk:

- HydroGym v1.0.1 Zenodo archive, MAIA solver image and pinned HF resource hashes
  listed in `README.md` / `STRICT_REPRODUCTION.md`.
- [Author TorchRL source](https://github.com/cl126162/TheHydroGymProject/tree/20a483ecc6ea711f6b380fff05f212ca4d7e2c67),
  commit `20a483ecc6ea711f6b380fff05f212ca4d7e2c67`.
  Network factories are loaded directly from this read-only checkout. PPO and
  TD3 also use its loss and optimizer factories. API-name compatibility edits
  and source SHA256 are saved in each run's `author-source.json`.

## Fixed setup and explicit reconstruction choices

| Setting | This experiment | Basis / limitation |
| --- | --- | --- |
| Re / CFD asset | 2D Pinball, Re100; pinned public MAIA fields and mesh | No field/mesh replacement or solver simplification |
| Reward | `-abs(total_drag)-abs(total_lift)` | Supplement omega=1; do not reproduce the old source's double-drag indexing bug |
| Actions / episode | 200, 170 CFD steps per action | Supplement; compensate native off-by-one horizon |
| Max physical control | 0.25 (explicit CLI argument) | Current HF package default. Old author YAML 0.5 was tested, not silently assumed compatible with newer assets; see diagnostic below |
| Sensors | 7x7, x=1.5…7.5, y=-1.5…1.5; u,v (98 inputs) | Author case-specific environment defaults; learner script instead has a 7x5 grid; exact figure layout unknown |
| Observation scaling | Fixed free-stream U_inf | Current official adapter; older example passes identity scaling; recorded difference |
| Training budget | 100,000 controls = 500 full training episodes | Paper plot and YAML episode count; YAML collector instead says 200,000 frames |
| Independent training seeds | 42, 43, 44 | Three as paper; exact published values unknown, only example seed 42 available |
| Networks | Separate actor/critic, 256x256, ReLU | Author widths; paper ReLU overrides older YAML Mish |
| PPO | gamma .99, GAE .95, clip .2, entropy .01, critic coefficient .2 | Paper GAE; typical paper clipping chosen; other values author YAML |
| PPO updates | Collect 400, 25 epochs, minibatch 25, separate Adam, LR 3e-4 linearly annealed, GAE recomputed each epoch | Author training schedule; no pilot's combined optimizer or global .5 gradient clipping |
| DDPG | BatchNorm in both hidden layers of actor/critic; tau .005, replay 100k, batch 256 | Supplement; author network factory extended with BN |
| DDPG update | MSE Bellman critic, then deterministic actor gradient, frozen target networks | Supplement equations 29/30; local update needed for explicit BN train/eval handling, not an unchanged author loss |
| DDPG BN targets | Eval mode, copy online running stats during target update; Polyak-average weights | Explicit implementation choice; exact paper BN behavior not published |
| DDPG exploration | OU theta .15, sigma .2, dt .01; epsilon 1→.1 over 1M controls | Paper numeric .15/.2; public TorchRL wrapper supplies dt/annealing defaults; paper calls .15 mu, convention ambiguous |
| TD3 | Twin critics, target noise .2 clipped .5, delay 2, tau .005 | Supplement; author TorchRL loss |
| Off-policy optimization | Adam 1e-3, 1000 random controls, 400-transition batches, 1 update/control after warmup; Gaussian exploration .1 for TD3 | Explicit defaults because no author per-case DDPG/TD3 YAML found |
| Evaluations | Frozen deterministic actor, every 4000 training controls and at initialization/end | Logging frequency differs from author's 800; samples never enter training/replay |

Only training seed changes across the three repeats. Native CFD reset reloads
the same initial flow: different seed labels on repeated evaluation rollouts
do **not** represent independent physical test samples.

## Artifacts and comparisons

Each run saves its source snapshot, resolved config, physics protocol, raw episode
records, learning updates, immutable `policy-NNNNNNN.pt` snapshots, `candidate.pt`
and `best.pt` (best raw deterministic return at the fixed evaluation schedule).
These are inference checkpoints, **not exact-resume CFD/replay state**.
`training_step` counts only training interactions. Evaluation episodes have an
explicit phase; never count them as extra training episodes.

Evaluation records include all per-step drag/lift, actions and native rewards.
Report separately: raw summed return, return per control step, full-episode mean
drag/lift, and last-half mean drag/lift, with matching uncontrolled windows.
The last-half window is fixed in advance. Also retain full rollout plots; do not
choose a favorable window or rescale rewards to force a match to a figure.
The paper plot's exact reward normalization remains unknown.

A successful process, falling training loss, or one favorable seed is insufficient.
First check unchanged uncontrolled physics, then independent-process frozen-policy
evaluation, then all three training seeds and learning curves. Compare each
algorithm separately, with mean/std and differences from the published setup.
Performance matching is distinct from proving exact unpublished-run provenance.

## Commands

Run the CPU learner wiring tests in the baseline image (see `test_public_learner.py`),
then short **native CFD** training/evaluation checks before full experiments.

```bash
bash baselines/run.sh hg-public-pinball2d-ppo-s42 GPU_INDEX train \
  --learner public-source --protocol author-paper --max-control 0.25 --algo PPO --seed 42 \
  --steps 100000 --eval-every 4000 --evaluate-published
bash baselines/run.sh hg-public-pinball2d-ppo-s42-eval GPU_INDEX evaluate \
  --learner public-source --protocol author-paper --max-control 0.25 --algo PPO --seed 42 \
  --checkpoint /runs/hg-public-pinball2d-ppo-s42/candidate.pt --episodes 1
```

Repeat with DDPG/TD3 and seeds 43/44. External evaluation of `best.pt` uses the
same protocol; it is a development-selected policy, not an unseen-test estimate.
Keep the prior 40k custom pilots under their original names for historical comparison.

`--evaluate-published` automatically starts fresh Python/MPI/MAIA processes after
training exits, in new CFD work directories, to evaluate final and best artifacts.
`independent-final/` and `independent-best/` hold those results; identical file
hashes are evaluated once. `workflow-completed.json` is written only after those
processes succeed. `completed.json` alone only describes the training stage.

## Control-range preflight, 2026-09-05

The old author YAML's 0.5 limit was initially tested. At 1,200 training controls,
the DDPG+BN actor produced a feedback trajectory that diverged in native CFD
(nonfinite observation/reward after rapidly growing forces). The check exited
with an error, retaining the policy and partial trajectory. PPO and TD3 short
checks completed at 0.5. This is **not proof that every 0.5 control is unstable**:
fixed all-negative and counter-rotating controls were finite at both limits.

The older source's domain, refinement, reduction factor, restart step and ramp
settings also differ from the current asset package. Copying its one scalar
actuator limit is not sufficient to reconstruct that old setup. For the main
performance reproduction, all methods use the current package's 0.25 limit,
explicitly recorded, while retaining paper reward/horizon and authored learners.
We do not mask solver divergence, inject finite replacement rewards, or change
the physical score. The 0.5 trials remain separately named diagnostics.

The paper-described counter-rotating mechanism is achievable with the current
package: fixed normalized action `[0,-1,1]`, max_control=0.25, produced full-window
mean drag 0.332960 versus zero 2.905470 (**88.54% reduction**), and last-half drag
0.248932 versus zero 2.905670 (**91.43% reduction**). Full return -72.431818;
RMS lift 0.044846. This is a **hand-set physical control diagnostic**, not evidence
that PPO/DDPG/TD3 training has reproduced the paper. It establishes a reachable
performance scale for the forthcoming learned-policy comparison.

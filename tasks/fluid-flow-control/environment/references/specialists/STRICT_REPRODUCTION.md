# Exact-paper reproduction audit — 2026-09-05

## Current decision (supersedes the original strict-only pause below)

The user has explicitly requested **no author contact**, performance-first
best-effort reproduction from the public paper and projects, with the closest
documented setup possible. Missing final configs no longer block local trials.
See [PUBLIC_REPRODUCTION.md](PUBLIC_REPRODUCTION.md) for the executable experiment,
known choices and acceptance criteria. No external message has been sent.

## Historical decision before that clarification

The completed 40,000-control-step custom trials are **not reproductions of the
paper's experiments**. Do not promote their scores or extend those learners and
call that an exact reproduction. The user requested identical experimental
setup, not a new implementation with inferred parameters. No new training has
been launched during this audit. Existing trial artifacts are preserved.

The original reproduction request is not complete. The next dependency is a
verified mapping from the final paper's figures to runnable author code,
configuration files, assets, seeds and plotting/evaluation conventions. No
external request has been sent to the authors.

## Newly located author source

- [TheHydroGymProject](https://github.com/cl126162/TheHydroGymProject/tree/20a483ecc6ea711f6b380fff05f212ca4d7e2c67),
  commit `20a483ecc6ea711f6b380fff05f212ca4d7e2c67`, contains the author's
  `torchRL_examples/ppo.py`, `ddpg.py`, `td3.py`, supporting utilities and several
  example YAML configurations. Its README explicitly describes an incomplete
  public release, predating the final Nature publication.
- Local untouched checkout:
  `/mnt/workspace/hydrogym/paper/source/TheHydroGymProject`.
- The main HydroGym repository and its Nature-linked Zenodo v1.0.1 archive do
  not include those `torchRL_examples`. The earlier search missed this separate
  author repository. Finding it corrects the earlier incomplete source audit;
  it does not establish that its example defaults generated the final figures.
- [Controlling-Kolmogorov-Flow](https://github.com/smokbel/Controlling-Kolmogorov-Flow/tree/45563a836b2eb3ad8ba319ec873b7b2df4802585),
  commit `45563a836b2eb3ad8ba319ec873b7b2df4802585`, contains author JAX PPO/RNN
  examples. The `eps` branch was also checked. No identified implementation of
  the paper's GPPO gradient-mixing rule was found. A standard PPO
  `jax.value_and_grad(loss)` is not evidence of differentiation through CFD.

## Concrete mismatches

The paper references below are the [Nature article](https://www.nature.com/articles/s41586-026-10917-6),
[Extended Data Fig. 2](https://www.nature.com/articles/s41586-026-10917-6/figures/6)
and supplementary sections 4A–4D and 5B. Local originals are `article.pdf` and
`supplementary.pdf` under `/mnt/workspace/hydrogym/paper/`.

| Item | Final paper | Completed custom pilot | Newly found author example |
| --- | --- | --- | --- |
| Pinball learning curves | Both 2D and 3D figure axes extend to 500 episodes; mean and SD of 3 training seeds | 200 episodes, seed 0 only | YAML says 500 episodes but collector uses 200,000 frames; the loop uses collector frames, so the actual figure-axis mapping needs confirmation |
| PPO activation | ReLU | ReLU | Mish |
| PPO GAE lambda | 0.95 | 0.95 | 0.97 |
| PPO clipping | 0.2 described as typical; exact per-case setting needed | 0.2 | 0.1 |
| PPO entropy regularization | Used | Omitted | Coefficient 0.01 |
| PPO optimization | Per-environment configuration | 10 epochs, minibatch 32, fixed LR, combined optimizer | 25 epochs, minibatch 25, LR annealing, separate optimizers |
| DDPG batch normalization | Used | Omitted | No explicit batch normalization in the published actor/critic construction; needs clarification |
| Pinball lift penalty | omega=1; second term uses total lift | omega=1, total lift | YAML omega=0; source incorrectly indexes drag for the second term |
| Exact probe layout | Illustrated; final numeric per-case list not supplied in supplement | Inferred 8x8 wake grid | Learner hard-codes a different grid from the environment defaults; correspondence to figure unknown |
| Pinball actuator scaling | Normalized action bounds alone do not fix physical actuator strength | Public-Hub 2D max_control=0.25 | Older YAML max_control=0.5 |
| Numeric reward comparison | Pinball plot reward scale is around -1.5 to 0 in 2D | Logged raw 200-step summed returns | Need original aggregation/normalization/plotting convention; values cannot simply be overlaid |

The old Pinball YAML is
[`config_pinball_ppo.yaml`](https://github.com/cl126162/TheHydroGymProject/blob/20a483ecc6ea711f6b380fff05f212ca4d7e2c67/maiaGym/envs/pinball/default_configuration_files/Pinball_2D_Re100/config_pinball_ppo.yaml).
It must not silently override the final paper when the two conflict.

## Verified execution blockers in the old example

1. The PPO entry reads `cfg.maia.nDim` and `cfg.maia.environment`, but the shipped
   Pinball YAML provides neither. Its `config_path` also points to a cylinder
   YAML. These are not a complete ready-to-run Pinball experiment configuration.
2. The 2D probe branch in all three learners references `probe_list` before local
   initialization. See [PPO line 74](https://github.com/cl126162/TheHydroGymProject/blob/20a483ecc6ea711f6b380fff05f212ca4d7e2c67/torchRL_examples/ppo.py#L74).
3. [Pinball reward line 30](https://github.com/cl126162/TheHydroGymProject/blob/20a483ecc6ea711f6b380fff05f212ca4d7e2c67/maiaGym/envs/pinball/pinball_environment.py#L30)
   uses force column 0 twice. A unit check executed the unmodified function with
   total drag 6, total lift 60 and omega=1: it returned -12, whereas the paper's
   formula gives -66. This is a code/formula discrepancy, not a measured CFD
   score. Whether the actual published runs used this version requires author
   confirmation; fixing it locally does not establish that provenance.
4. Six checked-in YAML files were found, all named PPO configs; no complete
   per-case DDPG/TD3 experiment YAML was found, including the eight-commit history.
5. `requirements.txt` names libraries without versions. In our existing
   TorchRL 0.13.3 image, unmodified `ppo.py --help` fails importing
   `SyncDataCollector`. Exact dependency versions used for the paper are needed;
   installing a guessed historical version would only create another assumption.

## Numerical evidence from the completed pilots

Independent deterministic-policy evaluation at 40,000 steps gave:

| Method | Raw return | Mean drag |
| --- | ---: | ---: |
| No control | -587.8810 | 2.90547 |
| PPO | -1107.9880 | 2.11056 |
| DDPG | -501.6032 | 1.52286 |
| TD3 | -264.3145 | 0.57508 |

TD3's 80.2% episode-average drag reduction is **not a match claim** for the
paper's approximately 90% reduction. The plot's rollout window and averaging
convention must also match. The paper shows all three algorithms learning
effective Pinball control, unlike this pilot's PPO outcome.

The complete 3D zero-control episode returned mean drag 2.71501, whereas
supplementary Table 3 reports 2.949 for 3D Re150 (about 7.93% lower here).
Initial flow checkpoint, averaging window and mesh/solver parity are unresolved;
successful solver exit is insufficient validation.

## Remaining information needed for an exact-provenance claim (not a blocker)

An exact-provenance claim would require the final-publication package for
Extended Data Fig. 2a (2D Pinball Re100) and Fig. 2a (3D Pinball Re150):

- Exact commit and command for each PPO/DDPG/TD3 run, fully resolved per-case
  configs, dependency versions/image digest, and all three training seed values.
- Exact meshes/restart states, physical actuator limits, sensor coordinates,
  normalization statistics, rewards, time steps, reset/termination semantics,
  collector budget, optimizer/exploration settings and any omitted patches.
- Raw per-seed learning curves and test-rollout data, plotting code and the
  definition of the episode/reward/time axes. The article says source data are
  provided, but no downloadable numerical source-data file was found in the
  inspected article/figure links; only the supplement and peer-review PDFs.
- The final GPPO implementation/configs for Fig. 3b, and corresponding packages
  for the cavity, cylinder and gust-airfoil comparisons before claiming full
  benchmark reproduction.

Once supplied, preserve the source unchanged and limit local adaptation to
documented host/container/path compatibility. First verify uncontrolled physics
and reward calculation, then replay any released reference policy, then retrain
with the original budget/seeds and compare identical metrics and uncertainty.
The user's subsequent instruction permits documented local reconstruction and
tuning without that package. Do not label those choices exact author settings.

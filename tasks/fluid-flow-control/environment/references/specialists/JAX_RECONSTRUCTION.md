# Differentiable Kolmogorov baseline audit

Latest audit (2026-09-06 15:48 UTC): R8 is the latest tested task image, including
Eq34, rollout40, actor export and resident replay. The first GPU5000 PPO run
completed; same-GPU independent evaluation returned27.94386697, with all-zero
deterministic actions. This numerical proximity to the plotted PPO endpoint
does not establish paper performance/setup equivalence or improved mixing.
The scaled5000 and public-rollout40/5120 comparisons remain active. Eq34's
CPU40 pilot passed80 updates and an independent raw-reward aggregation audit;
its full GPU run is queued. `--rollout-steps 40` matches the public example's
40x4 batch/minibatch20, unlike the earlier reconstruction's10x4/minibatch5.
See GOAL_PROGRESS.md for current run evidence. Historical notes below preserve
earlier states and must not be mistaken for the latest live GPU allocation.

The cited NeurIPS2023 GI-PPO has public source at
[SonSang/gippo](https://github.com/SonSang/gippo), inspected revision
`d7b741876d41d7e0a35556c303217c92cabc3a86`. Its alpha-policy update regresses
actions toward advantage-gradient-shifted targets before a modified PPO update.
This is not a recovery of HydroGym's undisclosed gradient/entropy-weighted
implementation. Do not silently relabel a port as the paper's exact GPPO.

## Full-policy derivative diagnostic — 2026-09-06 12:02 UTC

`jax_policy_gradient.py` now differentiates through the actual public actor,
feedback observations and consecutive CFD intervals without Gym auto-resets or
stop-gradient boundaries. It uses the frozen GPU80 checkpoint, FP64 arithmetic
for a derivative diagnostic, and the same Re250/64x64/.001/10-time-unit physics.
Pure `jax.checkpoint` rematerialization bounds backward-pass memory without
changing timesteps or forward RK arithmetic. The objective for this diagnostic
is TKE alone, so the direct action-cost derivative cannot masquerade as a
physical sensitivity. No training/update or GPPO-performance claim is made.

The first two-control check gave TKE3.4187503 and policy-gradient norm83.90117.
Central differences gave40.3684 at perturbation1e-5 (51.9% error) and83.3704
at1e-6 (.633% error). Its strict two-step-size check failed, and the failure
remains recorded. This rapid improvement suggests finite-difference truncation
in a sensitive trajectory, not yet an established gradient bug or a pass.
A new convergence check retains all coarser results and requires both1e-6 and
1e-7 differences to agree within2%; it also compares ordinary and value-gradient
forward objectives. It is running on CPU, not competing for CFD GPUs.

The native JAX solver is executable in the pinned runtime. This is not yet the
paper's PPO/GPPO performance reproduction. Do not present the public example's
default training return as the paper's mixing objective.

## GPU runtime audit — 2026-09-06 11:29 UTC

The CPU full-budget run remains active. Its immutable 360-control checkpoint was
independently evaluated: deterministic actions were all zero, giving exactly the
zero-control return 29.730408 and mean TKE 2.973041. This is not an improvement in
mixing, even though the numerical return resembles the paper's PPO curve scale.

The first GPU PPO check failed before publishing a training checkpoint, in
cuSolver QR used by the public orthogonal initializer. A minimal QR diagnostic
reproduced the error independently of HydroGym. Its process maps showed both
pip CUDA 12.8 cuBLAS and NVHPC CUDA 12.9 cuBLASLt/JitLink libraries loaded.
`jax_runtime.sh` now provides a JAX-only bootstrap: clear the inherited dynamic
library search path, load the pinned EasyBuild Python module, and use pip CUDA
libraries. It does not change MAIA's separate required runtime bootstrap.

Under this clean loader environment the internal cuSolver error disappeared.
The first post-cleanup numerical check was overly strict about a GPU default
precision matrix product, not QR itself. The corrected independent host-FP64
orthogonality check passes all three actor-layer shapes: max error 4.36e-7,
versus up to 1.89e-4 when checking via default GPU matrix multiplication.
Both diagnostic versions and their source/configuration remain recorded.

The clean-loader retry `hg-jax-gpu-ppo-clean-check-20260906` then exposed a
separate publication problem: `JAX_PLATFORMS=cuda` disables the CPU backend
needed by `jax.debug.callback`. It failed at first publication; its output is
retained. GPU runs now use `JAX_PLATFORMS=cuda,cpu`, and `jax_public.py` checks
CPU backend availability before spending time on CFD rollouts.

`hg-jax-gpu-ppo-dualbackend-check-20260906` passed 80 global controls, eight
episodes and 160 optimizer updates in 162.5 seconds. Parameter change L2 was
.5668, and the frozen checkpoint SHA is
`febd7c3355d45e5af4ebbcf3777ea357b2a53e7d356c9d264f71e07fc0dcad69`.
A fresh ten-control GPU replay returned 13.623970 with mean TKE 2.610145,
still below zero control (29.730408). This establishes usable training and
checkpoint replay, not paper-level performance. The full 5,000-control GPU run
`hg-jax-gpu-ppo-paper-budget-20260906` started at 11:35 UTC using that successful
pilot's frozen Python source and runtime bootstrap. CPU and GPU runs share the
same seed; they are backend comparisons, not independent seed replications.
GPU0 is shared with Pinball, so timing is not isolated-device performance.

The loader check follows the [official JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)
about inherited CUDA search paths. This is a local runtime correction, not a
change to the published PPO update or fluid objective.

## Verified diagnostic

`jax_probe.py` runs the actual pinned `KolmogorovFlow` solver, computes its TKE
gradient through a 10-time-unit control interval, and checks a directional
derivative against central finite differences. On CPU, 2026-09-06:

- Re200, 64x64, dt=0.001; native default initialization, four forcing amplitudes.
- TKE=1.3754824320 after action `[0.13, 0.07, -0.04, 0.02]`.
- Analytical directional derivative 0.02167736480; numerical 0.02167736522.
- Whole diagnostic 50.8 seconds, including compilation; no policy was trained.
- Output `/mnt/workspace/hydrogym/reproduction/hg-jax-gradient-diagnostic-20260906/diagnostic.json`.

This confirms a nonzero, finite physical gradient, not only differentiation of
an action penalty. It directly uses `step_env`; a future GPPO implementation must
also ensure its rollout wrappers preserve the solver's analytic gradient.

## Public-source differences that must be resolved explicitly

1. Extended Data Table 1 states 4,096 degrees of freedom for Kolmogorov (consistent
   with 64x64). Its Reynolds range is 40–500, not an exact Fig. 3b configuration.
   The current code defaults to Re200. The older related author's example uses
   Re250 and 256x256; the Hub YAML names Re40. These are not interchangeable.
2. Native `_reward` is `-(reward_alpha * TKE + sum(abs(action)))`, with
   `reward_alpha=0` by default. Thus default PPO optimizes actuation alone and
   cannot establish the paper's enhanced-mixing result. A positive coefficient
   also minimizes, not maximizes, TKE. The older related example maximizes TKE
   minus 75 times nonnegative action amplitudes; exact figure weighting is unknown.
3. Public PPO's default budget of 100 transitions is below its 160-transition
   update batch. A runnable baseline must set a real budget with actual updates.
4. Supplementary Section 4D describes adding an analytic trajectory-gradient
   objective to PPO, and separately discusses a gradient-augmented reward. The
   article's Fig. 3b uses the former. No original GPPO trainer/weighting was located
   in the pinned platform, archived release or related public author repository.
   Any implementation here must be identified as a public-description reconstruction,
   with weighting, gradient horizon, physical config and sample-cost accounting.
5. The paper says the Fig. 3b comparison illustrates one training seed. The
   three-independent-seed convention for PPO/DDPG/TD3 Fig. 2/Extended Fig. 2 must
   not be incorrectly attributed to this particular plot.
6. The related older `jaxrl/flow_env_gymnax.py` computes observation and physical
   reward from the **incoming** state before returning the updated flow state.
   The pinned HydroGym class instead evaluates both on the **new** trajectory.
   Our reconstructions retain the current class's transition timing; borrowing
   the older TKE-minus-control formula is not full environment equivalence.
   The older example also samples eight specified probes, versus the current
   8x8 array. No figure-specific observation/lag configuration is established.
7. Both inspected integrators use floating-point floor division for step counts.
   For nominal action_time10, save_time1 and dt.001, Python computes
   int(10//.001)=9999 and int(1//.001)=999; ten saved blocks therefore advance
   9990 CFD steps (9.99 physical units), not exactly10000. This source behavior
   is retained, not silently fixed in existing experiments. Their metadata's
   action_time10 is the configured nominal interval, not a measured step count.

The article states source data accompany it, but the currently linked attachments
are only the supplementary PDF and peer-review PDF. A separate machine-readable
learning-curve attachment was not found in the page links. Do not invent exact
curve coordinates from the raster plot or claim their aggregation is known.

## Numerical failures and explicit correction experiments

The first public-source PPO mixing pilot became nonfinite in its second rollout.
Instrumented 80-transition FP32 and FP64 runs both confirmed this: the first
40-transition rollout and learner update were finite; the second batch contained
nonfinite CFD metrics and learner values. Higher precision alone did not resolve
the issue. FP32/FP64 random draws differ, so they are not identical-action replays.
Each failure retains its source/configuration/error record; no failed checkpoint
is published as a usable baseline.

Source inspection and executable tests found two concrete issues:

1. `utils.dealiasing` discards both `.at[...].set(0)` return values. JAX arrays
   are immutable, so the public function returns its input unchanged. A direct
   64x33 complex-array test confirmed this, both in the current platform and the
   older related author implementation. The optional correction applies a
   symmetric two-thirds spectral mask, preserving |kx|<Nx/3 and ky<Ny/3.
2. Current `PseudoSpectralNavierStokes2D.control_term` omits the `2*pi*i` factor
   for velocity-force-to-vorticity curl. The author's older `equations/base.py`
   includes it, as does the current class's own background `forcing_term`.
   The optional correction restores that factor. Its analytic sinusoidal-force
   unit test has maximum error 3.73e-14 in FP64.

`jax_public.py --solver-corrections none` remains the untouched solver path.
Separate `dealias` and `dealias-curl` runs change only these recorded numerical
choices, reusing the same public PPO updates. `jax_corrections.py` and source hashes
are saved with corrected runs. At this stage these are causal diagnostics, not
proof of resolved stability, paper-level learning or a faithful final-paper solver.
The native finite-difference gradient check above does not certify the correctness
of these physical discretization choices.

JAX update semantics: [official array.at documentation](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

### Imported-source guard and excluded organizer runs

Four newly added diagnostic launches accidentally omitted `PYTHONPATH=/upstream`,
importing the base image's older editable checkout rather than the mounted pinned
solver. Their results are excluded, not counted as reproduction evidence:
`hg-jax-ppo-{trace,dealias,curl}-f32-20260906` and
`hg-jax-corrected-trajectory-gradient-20260906`. The older integrator does not pass
the explicit control field, invalidating the latter's zero-gradient result.
The two active mismatched-version jobs were stopped; artifacts/logs were retained.

The original one-control gradient check and original FP32/FP64 failure diagnostics
did set the pinned path and remain valid. All subsequent JAX scripts use
`jax_source.py` to verify actual imported environment, integrator, utility and
wrapper file hashes before execution. A deliberate wrong-source test fails as
required. The replacement runs are prefixed `hg-jax-pinned-`; their metadata
records the verified imported paths/hashes. Full exclusion record:
`/mnt/workspace/hydrogym/reproduction/jax-source-exclusions-20260906.json`.

## Verified pinned-solver causal checks

All three replacement 80-transition runs used verified pinned imports. The
original solver again failed in the second rollout, while both `dealias` and
`dealias-curl` completed 160 optimizer updates / eight training episodes with
nonzero parameter changes. The latter's four final stochastic training returns
were -902.46, -1089.91, -866.33, -771.96; these are early negative returns, not a
paper performance match.

The failure trace localizes nonfinite CFD to control 2 in environments 1 and 2
of the second rollout, before observations/actions become nonfinite at control 3.
Replaying environment 1's exact two finite actions without any learner reproduced
the failure at control 2 in the original solver. The same actions stayed finite
with either correction mode. This supplies direct causal evidence for the
dealiasing stability correction, rather than comparing unrelated policy outcomes.
Action replay outputs: `hg-jax-actions-{native,dealias,curl}-20260906`.

The pinned `dealias-curl` two-control physical-gradient check also passed at
Re250/64x64: directional derivative 12.54131 analytically versus 12.41588 by
central difference (epsilon 1e-4), about 1% apart. Earlier controls have much larger
gradients than later ones; long-horizon gradient stability is not established.

A single-seed 5,000-transition / 500-episode PPO reconstruction is now running in
`hg-jax-ppo-paper-budget-v2-20260906`, using `dealias-curl`, the related-author
mixing objective and the documented grid/Re choices. Parameters and progress
are atomically published after each update using ordered callbacks; the console
now reports global control steps across all four environments. This is still not
the original undisclosed Fig.3b configuration. The first startup of this long run
was stopped before its first published checkpoint to make callback publication
ordered; its separate output directory is retained, not counted as a trial result.

## Subsequent runtime, gradient and artifact checks (2026-09-06)

- `jax_runtime.sh` selects a consistent Python/CUDA runtime without the MAIA
  module stack's competing CUDA libraries. GPU training requires
  `JAX_PLATFORMS=cuda,cpu` because checkpoint publication is a CPU callback.
  A GPU 80-control PPO check passed real updates, and a 5,000-control run is
  active. A separately labelled optimizer reward-scale=.01 trial is queued.
  All scores retain the original physical reward units.
- The two-control full-policy/CFD derivative passed convergence checks at
  eps=1e-6 and1e-7 (0.633% and0.0063% error). The full ten-control derivative
  did **not** pass: gradient norm1.7552e13 and ordinary/value-and-grad forward
  TKE differed by about0.00365. That run remains excluded, not a GPPO baseline.
- The supplement explicitly also describes Eq.34's alternative: augment an
  instantaneous maximized reward r with `(dr/da).a`, differentiating a full CFD
  control interval while holding its initial state fixed. `jax_gradient_reward.py`
  implements this explicit formula with coefficient1, not an invented claim
  about the unpublished adaptive Eq.32 gradient weights. Two actual FP64 CFD
  intervals passed ordinary/gradient forward parity and central differences of
  the physical TKE gradient at1e-5 and1e-6. This verifies physics sensitivity
  beyond differentiating the direct action penalty. Output:
  `hg-jax-eq34-gradient-check-20260906` (156.5seconds).
- `jax_public.py --gradient-reward` is labelled `PPO-Eq34-reconstruction`.
  An actual 40-control pilot is active; it is not yet an accepted trained result.
  Its optimizer consumes augmented rewards, but logged episode returns are
  reconstructed from **unaugmented physical rewards**, with checked episode
  alignment. Neither this alternative nor its coefficient is asserted to be
  the original Fig.3b implementation.
- `export_jax_policy.py` produces a standalone actor-only Flax resident package.
  The GPU80 pilot's exported actor matched64 input checks exactly. Same-CPU
  direct and resident-protocol CFD replays then matched all trajectory fields
  exactly: return12.735181093215942, package hash
  `d9421eb28f2315fe06edda1139d9c03d5e8bbb631ba73e02ca9d0ea3564c0f52`.
  The older GPU replay returned13.62397027015686; do not pool different backend
  trajectories or confuse them with independent training seeds. Policy transport
  is validated, not paper performance. `jax_evaluate.py --package` performs this
  relocation/replay and requires the production `--evaluator-code` path.

These newest sources postdate image r4; rebuild/test the development image before
calling them shipped in the in-image starter. Running jobs use saved per-run code.

## Completed 500-episode GPU PPO and independent replay — 2026-09-06

`hg-jax-gpu-ppo-paper-budget-20260906` completed 5,000 native controls,
500 episodes and 10,000 optimizer updates in 8120.08 seconds. Checkpoint
SHA256 is `446ec468b77d327012b223ce764e9f28e01aa7393b08dc5ef47db27d626ed1ae`.
This used rollout10/minibatch5; the related author's rollout40/minibatch20
is now running separately for 5,120 controls (512 episodes, explicitly 2.4%
over the nominal 500-episode budget).

Fresh direct and resident-protocol GPU2 evaluations of the 5,000-control
checkpoint both returned 27.943866968154907. Actions, rewards and TKE matched
exactly, but observations differed by at most 4.76837158203125e-7: the entire
trajectory is **not bit-identical**. The zero-policy replay returned
27.943866848945618. The trained deterministic policy takes zero control, so
this does not establish enhanced mixing. CPU replay returned 29.7304080725;
backend differences must not be counted as independent training seeds.

The GPU score is numerically near the visually read PPO endpoint around 28
in Fig.3b, but deterministic replay is not the stochastic training curve and
the original complete configuration remains unrecovered. This is not GPPO's
approximately 45 endpoint and is not full paper certification. The measured
trajectory differences are recorded in `jax-5000-gpu2-parity-audit-20260906.json`.

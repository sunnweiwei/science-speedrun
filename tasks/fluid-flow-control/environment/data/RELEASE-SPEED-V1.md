# Historical first-tested 28-case release

This document preserves the previous speed-v1 selection. See [RELEASE.md](RELEASE.md) for the current selection; expansion review is documented separately.

# HydroGym public speed-aware release v1

The membership and source-data release is `hydrogym-public-speed-v1`.
The [manifest](release-speed-v1.json) is authoritative; the [selection](selection-speed-v1.json)
and [measured evidence](cost-evidence-speed-v1.json) explain every decision.
This is our cost-aware subset, not the complete original paper benchmark.

Execution readiness is separately **data-release-not-cfd-certified**. No
complete evaluated benchmark score, full-episode speed guarantee or paper-policy
reproduction is claimed by these short probes. The executable `suite.json`
is not published yet.

## Size and membership

The cap is **500,000,000,000 bytes (500 decimal GB)** for one shared copy of
fixed training/evaluation source assets. It is an upper bound, not a target to
fill. It excludes container images, download caches, generated experience,
checkpoints, logs and solver scratch. Interactive training has no fixed offline
transition count.

Frozen 2026-09-07: **28 cases = 19 two-dimensional + 9 three-dimensional**, with
**390 source files / 138,268,990,191 bytes (138.269 GB)**. Source storage falls
from the historical 488.771 GB to 138.269 GB; no selected case's inputs are cut
down. There are 26 MAIA, one JAX and one JAX-Fluids case.

| Group | Cases | Source GB | Short s/control on one H100 |
| --- | ---: | ---: | ---: |
| Cavity 2D Re7500 | 1 | 0.057 | 0.159 |
| Cylinder 2D Re200/1000/3900 | 3 | 0.431 | 0.112–0.687 |
| Square cylinder 2D Re200/1000 | 2 | 0.157 | 0.096–0.382 |
| Pinball 2D Re30/75/100/150 | 4 | 0.276 | 0.208–0.394 |
| NACA0012 2D, gust/no-gust | 8 | 0.406 | 0.092–0.449 |
| Coarse nozzle 2D | 1 | 0.008 | 0.904 |
| JAX turbulent channel 3D | 1 | 0.009 | 0.104 |
| Pinball 3D Re30/75/100/150 | 4 | 33.369 | 10.126–15.947 |
| Sphere 3D Re300 | 1 | 19.965 | 28.752 |
| NACA0012 gust 3D: Re100 AOA20/40, Re1000 AOA20 | 3 | 83.591 | 5.560–18.173 |
| **Total** | **28** | **138.269** | |

Exact retained 3D identities and startup costs:

| Case ID | Source GB | Short s/control | Initial / second-reset s | Post-warmup samples |
| --- | ---: | ---: | ---: | ---: |
| Channel_3D_Retau180 | 0.009 | 0.104 | 0.408 / 0.004 | 6 |
| NACA0012Gust_3D_Re1000_AOA20 | 36.490 | 18.173 | 790.9 / 49.7 | 2 |
| NACA0012Gust_3D_Re100_AOA20 | 23.608 | 5.560 | 539.2 / 25.4 | 5 |
| NACA0012Gust_3D_Re100_AOA40 | 23.494 | 6.255 | 531.4 / 26.6 | 5 |
| Pinball_3D_Re100 | 8.342 | 10.126 | 243.6 / 17.9 | 5 |
| Pinball_3D_Re150 | 8.342 | 10.642 | 250.1 / 18.1 | 5 |
| Pinball_3D_Re30 | 8.342 | 15.947 | 247.8 / 21.2 | 5 |
| Pinball_3D_Re75 | 8.342 | 12.157 | 241.4 / 19.4 | 5 |
| Sphere_3D_Re300 | 19.965 | 28.752 | 131.6 / 44.5 | 5 |

Initial/reset measurements exclude outer container/workspace preparation.
The three gust-wing resources are new relative to the old 47-case selection.
The Re1000 wing first hit a 900-second probe limit after measuring controls;
its follow-up completed three controls (including a nonzero action), a second
reset and clean exit within 891 seconds using a 1,200-second diagnostic cap.
No CFD interval was shortened. The two independent attempts agree on roughly
18–19 seconds/control, but only two post-warmup samples in the complete attempt
are used in the frozen timing mean.

We reviewed all **62 resource-complete variants** in the pinned source inventory,
not just the old 47-case list. All 47 previous cases received bounded diagnostic
attempts; six additional 3D variants received new short probes. Every unselected
complete resource has an explicit reason. Unmeasured cases are not labelled
slow merely because their files are large.

The retained 2D set is complementary, not a substitute for 3D:

- `Cavity_2D_Re7500`.
- `Cylinder_2D_Re200`, `Cylinder_2D_Re1000`, `Cylinder_2D_Re3900`.
- `SquareCylinder_2D_Re200`, `SquareCylinder_2D_Re1000`.
- `Pinball_2D_Re30`, `Pinball_2D_Re75`, `Pinball_2D_Re100`, `Pinball_2D_Re150`.
- The eight 2D NACA0012 variants: gust/no-gust × Re100/Re1000 × AOA20/AOA40.
- `Nozzle2D_coarse`, retaining a compressible shock-vector-control problem.

The 3D set preserves different phenomena: turbulent wall flow, interacting
multi-body wakes across Reynolds numbers, a sphere wake, and airfoil gust
control across incidence/Reynolds number. Lower-Re 3D cases are not advertised
as replacements for high-Re industrial turbulence.

## How the cost screen was applied

On this host, retained cases must have a completed reset/control/reset probe,
clean process exit, and at least two post-warmup control samples averaging no
more than **30 seconds per native control action on one H100**. This is an
organizer selection rule, not an agent timeout or an evaluator timeout.

MAIA and JAX-Fluids timings exclude the first control; JAX timings exclude the
first two. The evidence includes actual samples, initial/reset costs, effective
configuration, image identity and raw-record hashes. These are short estimates
under shared-host conditions, not confidence intervals or guaranteed throughput.
MAIA uses one Python rank plus one solver rank; repeated sampled controls are
not independent full episodes.

Different native controls advance different amounts of simulated time.
Consequently this table cannot establish that all training will finish in two
hours. For example, 1,000 sphere controls alone extrapolate to roughly eight
H100-hours at the short measured rate, before resets or learner work. Eight GPUs
allow concurrent cases/environments but do not remove an episode's sequential
dependency. Full-suite replay and sustained training costs remain to measure.

No restart truncation, mesh coarsening, time-step change or shorter native
actuation interval was used to make the retained cases faster.

### Explicit configuration boundary

For MAIA, timing uses the pinned native YAML action scaling, reward and control
interval. Sensors are explicit diagnostic coordinates, **not recovered paper
sensors or a certified evaluation observation map**. The original source YAML
and effective timing configuration are both preserved.

For `Channel_3D_Retau180`, the measured profile is the pinned
`ChannelFlowSpectralEnv` class default: **72³ grid, 24 actions, 50 CFD steps per
action, dt=0.0002**, loading the provided initial fields. Its asset YAML differs;
we do not silently describe this timing as a YAML-profile measurement. The
executable evaluation protocol must explicitly bind and review the class
profile, including its observation, reward and horizon. The
[official channel example](https://dynamicslab.github.io/hydrogym/docs/examples/jax/channel/)
describes the differentiable 3D channel family; our exact timing identity is the
pinned code and recorded parameters, not a changing documentation default.

The coarse nozzle probe exercised native JAX-Fluids with finite observations,
rewards and primitive fields. This is not full physical/reward validation.
JAX/JAX-Fluids production policy adapters still need integration.

## Exclusions are not all the same

Measured expensive candidates:

| Case | Source GB | Short seconds/control | Decision |
| --- | ---: | ---: | --- |
| Sphere 3D Re3700 | 34.44 | 171.26 | Exclude |
| Square cylinder 3D Re200 | 25.94 | 63.67 | Exclude |
| Cube 3D Re300 | 14.09 | 43.95 | Exclude |
| NACA0012 3D Re100 AOA20, no gust | 23.61 | 34.30 | Exclude |

These four rates each have only one post-warmup sample because the bounded probe
stopped early; they are screening estimates, not precise sustained rates.
Square-cylinder cleanup also hit its wall limit. Its measured step cost and
incomplete cleanup are both recorded rather than conflated.

Other exclusions have different evidence:

- Several high-Re airfoil/cylinder cases reached the initial-reset wall limit
  without steady-state timing. Their cost is **not established**, not reported
  as a measured slow solver. Cube Re3700 aborted during initialization.
- DRA2303 jet and coarse 3D nozzle have unresolved required configuration fields.
  Fine 3D nozzle has distributed restart/executor binding still to resolve.
- TCFmini's installed solver/restart configuration disagrees with its YAML.
  Its small CPU timing is not evidence for the intended configured task.
- Firedrake probes used one CFD substep per diagnostic interaction; the actual
  per-case RL control interval is unbound. These are not comparable native-action
  timings and are excluded on protocol readiness, not file size.
- Kolmogorov's timed class default has `reward_alpha=0` and parameter/objective
  discrepancies. A meaningful accepted control objective remains to bind.
- The fine 2D nozzle's short observed reward and field summaries did not change,
  including after a small action. This does not prove its solver is broken, but
  it does not yet establish a useful control response.
- Other resource-complete variants without an adequate compatible probe are
  explicitly `speed_not_established`.

The prior paper/resource audit's genuinely missing, marker-only or mismatched
inputs remain separately recorded in `release.json`. No missing geometry,
initial state or multi-agent actuation is synthesized or relabelled.

## Training, delivery and evaluation

Training interacts with exactly the selected environments and their supplied
initial-state pools. Every selected source file is preserved unchanged. The
environment restart checkpoints are CFD state, not pretrained policy weights.
Unselected archive environments are not additional training data.

Evaluation covers **all the same selected cases**, using **one jointly trained
checkpoint**, with no model or learned-normalizer updates during scoring.
Task-conditioned shared models are allowed; packing separately trained experts
into one file is not. The existing
[`hydrogym-policy/v1` contract](../contract.md) is unchanged.

This is known-task evaluation, not a private-task/zero-shot split. A different
seed label on an unchanged deterministic restart is not a new held-out flow.
Per-case primary measurement is mean undiscounted native episode return.
Different reward units are not blindly averaged. Normalized tracking requires
measured same-protocol anchors; historical paper curves do not become new
release scores automatically.

The evaluator requires an organizer-reviewed `suite.json` bound to the exact
new manifest hash and all selected identities. Sensors, actuation, rewards,
initialization, episode termination, horizons and solver launch settings must
be reviewed and validated. A missing adapter or failed CFD replay must not turn
the full benchmark into a silently smaller suite. Raw-return replay removes
only the reference-anchor requirement, not physical acceptance.

## Materialization and version preservation

From the task repository root:

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-speed-v1
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-speed-v1 --download
```

The first materializer call is a plan with no download. The download uses the
existing `hf` interface at pinned resource revision
`cf5ef6748443ae72e9647f398487e98d421a7243`, checks every Git/LFS hash, and writes
a receipt. `--verify-only` works offline. No unvalidated `suite.json` is created.

To reuse a local source archive on the same filesystem:

```bash
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-speed-v1 \
  --from-local /large-disk/existing-hydrogym
```

Only selected files are hard-linked and all bytes are verified. Both originals
and release views must remain immutable; mount only the selected root read-only
at `/data` using `TASK_DATA_DIR`. Keep generated work elsewhere. On a fresh
machine only selected files are downloaded; the full 1.09 TB archive is unnecessary.

The host target is
`/mnt/workspace/hydrogym/releases/hydrogym-public-speed-v1`.

On 2026-09-07, all **390 files** passed complete offline Git/LFS hash
verification in this new view. All are hard links to the unchanged full source
archive; only the exact 28 selected environment directories are present. The
receipt binds manifest SHA256
`23c53782f87b91c9be8babee3cab5e7cd16730c9dc79eb46d1b19c796be82890`.
It explicitly records `benchmark_validated: false`; no `suite.json` was created.
The current source also passed all **73 task tests without skips** in an offline,
CPU-only R9 container with the repository mounted read-only. This test run did
not rebuild the task image or perform full-suite CFD replay.

The old 47-case / 488.77 GB release remains unchanged in
[`release-500gb-v1.json`](release-500gb-v1.json), with its
[historical specification](RELEASE-500GB-V1.md) and original paper evidence.
Pass `--release fluid-flow-control/environment/data/release-500gb-v1.json` to materialize
that historical version into a different destination. Existing data roots and
running specialist experiments are not removed or retargeted.

Git contains only small code/metadata, not the source data. This update reuses
local upstream assets through the existing HF materialization interface; it
does not upload data or push a repository. Source asset/solver licenses still
apply. Before a formal AutoResearch session, the remaining work is complete
native protocol binding, full portable-model replay, and reference measurement.

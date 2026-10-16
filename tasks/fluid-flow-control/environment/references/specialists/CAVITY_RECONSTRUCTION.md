# Cavity reconstruction audit

## Missing 3D geometry reconstruction check — 2026-09-06 18:35 UTC

The public Hub revision remains unchanged and supplies no3D cavity resource.
The supplement (PDF pages32–34, section5C) supplies Re7500, span2.25D,
spanwise periodicity,375 substeps per3D action and the Eq.36 target prescription.
However, its prose domain is2.5D by1.5D by2.25D, while the paper's Extended
Data Table2 gives2.5H by1.25H by2.25H and34,990,848 degrees of freedom.
The supplied2D geometry instead extends x=0..4.5,y=-1..0.5 with a unit-depth
cavity at x=1.2..2.2. These are not one unambiguous extrusion recipe.

Likewise the stated spacing formula2.5D/2^12 is not1/227 of a cavity depth,
despite the adjacent approximate227-cells/depth statement. Refinement patches,
the exact3D actuator/sensor layout and the original restart/targets are still
not recovered. A generated mesh could be an explicitly reconstructed research
case, but cannot currently be represented as the paper's original3D baseline.
No mesh generation, source-resource replacement or invented release anchor was
performed on the basis of this incomplete specification.

## Explicit velocity diagnostics — 2026-09-06 18:40 UTC

Optional `cavity_train.py --record-probes` records the raw lattice u/v vectors
before policy normalization. Eq.36 reward and the policy's observation are
unchanged. `cavity_velocity_metrics.py` uses a complete declared episode and
reports every probe's temporal RMS about its own episode mean, normalized by
freestream velocity Ma/sqrt(3). It reports both the spatial mean and spatial RMS
of those per-probe values, explicitly named; neither is asserted to be the
paper's unspecified plotted u-prime aggregation. Raw returns remain unscaled.

Fifteen reward/archive/trajectory/grouping/velocity tests passed on host.
These additions postdate R9 and do not hot-change any live learner. CPU-only
zero100 validation uses separately frozen sources and the portable calibration
bundle. The first attempt stopped before observations because zero mode imported
TorchRL from a CPU environment lacking tensordict. The import is now confined to
train/evaluate modes; the preserved failed attempt has a separate exclusion
audit. `hg-cavity2d-cpu-zero-probe100-v2-20260906` is running on CPU16–17,
one solver rank,16GiB limit, no GPU or network. Scientific performance remains
unaccepted pending full native results and matching measurement definitions.

## Current experiments — 2026-09-06 16:14 UTC

The author34 calibration and Re4200 restart are complete and available together
under `data/reference-calibrations/Cavity_2D_Re4200-author34-v1`. A native run
with only the data mount reproduced the earlier zero100 return exactly
(-6802.668616663734). Three corrected1200-control PPO/DDPG/TD3 pilots completed
with independent replays; none establishes the paper's performance.

Corrected public PPO seed42 is running100000 controls. Its16k evaluation returned
-133649.82490, worse than zero control and its initial policy. Critic losses of
order1e8 motivate an explicitly separate optimizer comparison, not a proved bug
or a reason to stop that full-budget run. The existing `ppo-scaled` recipe uses
training reward x.1 and criticLR.001 instead of.0003; all CFD settings, sensors,
targets, restart, raw evaluation rewards and seed remain unchanged. These are
two optimization changes, not an isolated one-factor causal experiment.

The actual R8 learner passed four cavity-size update/reload tests, including
this recipe and time-limit bootstrap (`hg-cavity-scaled-learner-check-20260906`,
exit0). The test extension postdates R8 and was mounted separately; learner
code came from R8. Queue `hg-cavity-scaled-gpu2-20260906` waits for GPU2's current
JAX and shared-native checks, then runs1200 controls plus independent final/best
replays before a fresh100000-control trial. Inputs use the portable data bundle,
not historical run-directory paths. No current job is preempted or hot-modified.

Historical audit entries below describe earlier experimental states.

## Runtime correction — 2026-09-06 11:52 UTC

The first actual PPO check, using the public example's maximum control .075,
became nonfinite at training control22 before any optimizer update. All68 raw
probe values were nonfinite; normalized action was -.845611 (physical-.063421).
This followed a finite100-control initial deterministic-policy replay. Native
reward validation correctly rejected the state, but ordinary `finally: close()`
then hung in an MPI timestep send because the reward exception interrupted the
data-exchange phase. Read-only py-spy inspection confirmed this precise stack.
The hung check was stopped and preserved; a diagnostic rerun reproduced the
nonfinite transition and exited nonzero promptly after error logging/MPI abort.

`cavity_train.py` now records failed transition context and aborts paired MPI
ranks on errors before attempting normal close. Successful runs retain normal
shutdown. This does not suppress invalid CFD or turn it into a reward penalty.

The current pinned resource YAML sets max_control=.02, unlike the older example's
.075. Both are explicit CLI choices, recorded in provenance and independently
replayed with the same limit. A .02/400-control PPO diagnostic is now running.
This is a documented public-resource configuration alternative, not a claim
that the final paper used .02 or proof that the limit alone caused instability.
The full34-probe zero-control calibration continues unchanged on GPU6.

The paper's supplementary section 5C defines reward Eq.36 as a negative sum of
squared probe deviations standardized by target standard deviations. Its target
mean is estimated over 1,000 instability cycles. The current public native class
instead uses absolute deviations divided by observation scales. The older author
class agrees with the squared formula. Its default probe x coordinates 26..42
cannot be copied literally into the current geometry. A later geometry-unit audit
recovered the length conversion and public example target vectors, described
below; these are not verified figure-specific targets.

The pinned public 2D resources have Reynolds numbers 4140 and 7500; the paper's
2D learning demonstration is labelled 4200. Both public grid headers contain
213,940 cells, matching the paper table. However, their geometry is x=0..4.5,
y=-1..0.5 with cavity x=1.2..2.2, while supplementary prose describes a different
origin/domain length. Exact equivalence has not been established. No 3D cavity
resource was found in the pinned public release.

`cavity_probe.py` is a first uncontrolled measurement, not a learned baseline:

- Public Re4140 physics/restart/mesh remain unchanged, with paper's 400 CFD
  substeps per action; the long continuous horizon is explicitly diagnostic.
- A reconstructed 5x5 u/v probe array lies within the actual cavity fluid region
  and shear layer. It is not asserted to be the paper's exact sensor list.
- Raw unnormalized velocity traces, the current native reward, published mean
  samples, tail mean/std, spectral period and peak-count diagnostics are retained.
- The Feather resource contains only coordinates, rho, u, v; it does not supply
  target standard deviations or evidence of a 1,000-cycle averaging duration.
- No 1,000-step diagnostic is called a 1,000-cycle calibration, and no native
  absolute-deviation score is compared as if it were paper Eq.36.

The development image now installs pinned pyarrow to read the actual public
Feather target data. A subsequent long zero-control calibration and explicit
Eq.36 adapter will be necessary before paper-comparable learned-control trials.

The first native launch exposed a missing `maia.U_inf` in the published cavity
YAML. The measurement wrapper now initializes native normalization as `none`,
then explicitly derives the observation scale Ma/sqrt(3) from the actual solver
Mach number, consistent with native lattice-force normalization. This change is
recorded in `protocol.json`; source data are not edited. Raw velocity statistics
are collected before observation normalization.

## Initial measurements and Re4200 reconstruction

The complete Re4140 zero-control pilot produced 1,000 finite samples in 389.9 s.
The last 500 samples contain 74 peaks at the most variable v probe; its dominant
spectral period is 6.7568 control steps. Every probe has nonzero measured variance.
The maximum difference between the pilot's tail mean and the published Feather
mean is 0.010423 in raw lattice velocity, so the provided mean must not simply
be assumed interchangeable with a newly calibrated mean/std pair.

`derive_cavity4200.py` constructs `data/derived/Cavity_2D_Re4200`: only the runtime
Reynolds property and YAML Reynolds field change from 4140 to 4200. All heavy
assets remain relative links to the verified source release; `derivation.json`
records both generated hashes and the unchanged Re4140 initial restart/mean.
The public source and `materialized.json` are not rewritten to pretend this is an
original published environment. The native solver's `maia_res.txt` explicitly
confirms Reynolds number 4200 for the derived run.

A 16,000-control zero-action trajectory is running to allow relaxation and obtain
at least approximately 1,000 instability cycles in its last half, according to
the measured spectral period and observed peak counts. Both counts and blockwise
mean drift will be reported; an inadequate duration cannot be labelled calibrated.
This remains a reconstructed target, not verification of the paper's undisclosed
exact sensors, restart, target values or geometry. Before learned-control trials,
we also need a reproducible relaxed Re4200 solver restart and an Eq.36 adapter.

## Recovered author example and restart writer — 2026-09-06 11:08 UTC

The older geometry uses referenceLength=20 and domainLength=90. Current assets
use referenceLength=1 and domainLength=4.5, with the same major cavity corners
after division by 20: old x=[24,44], y=[-20,0] becomes x=[1.2,2.2], y=[-1,0].
The leading-edge actuator center similarly maps 23.5 to 1.175. One downstream
wall segment junction moves from 59/20=2.95 to 3.0; exact grid/paper equivalence
is still not asserted. This scale conversion was missing from the initial audit.

The author's `maiaGym/maia_env_core.py:main` contains a 25-point trapezoidal
shear-layer array plus nine near-wall probes, and four complete 68-entry vectors
for observation/target mean and scale. `cavity_author.py` reads those vectors
from the SHA-verified source and reconstructs the exact probe ordering, dividing
coordinates by 20 while leaving lattice velocities unchanged. All 34 mapped
probes lie in fluid, x=[1.3,2.19], y=[-0.25,0.075]. Native extraction confirms
68 observations and max nearest-cell distance 0.002236. We no longer treat these
public vectors as unavailable merely because the unconverted coordinates differ.

`hg-cavity2d-author34-zero-check-20260906` collects 1,000 native zero-control
samples with these author target vectors and actual Eq.36. This is not 1,000
cycles. `hg-cavity2d-author34-calibration-20260906` now collects 16,000 samples
on GPU 6, including the raw velocities needed to assess target mismatch and
construct a new target if required. It replaces the incomplete, underperforming
Pinball DDPG-slow/.25 ablation; all partial run artifacts remain, with a separate
reallocation audit. The earlier 25-probe calibration continues on GPU 1 as a
distinct reconstruction; their sensor statistics must not be interchanged.

`cavity_reward.py` implements Eq.36 and rejects incompatible sensor/Re/scale
bindings. Four numerical tests pass. `cavity_calibrate.py` only freezes completed
measurements with at least 1,000 estimated tail cycles, positive variances and
an explicitly recorded organizer block-mean drift criterion (default 0.25 std).
Its calibrated population mean reward is necessarily minus the observation
dimension; this identity must not be hidden by scaling scores to fit a plot.

The native restart writer overwrites `out_lb/restart_.Netcdf`, retaining a backup.
Our check first replaced the input symlink with a private copy, so published
assets stayed unchanged. With restartInterval=600000, it wrote a periodic state
at 600000 and a final state at 605002, both Re4200/Ma0.1. The latter's SHA is
`140d7b7bb2bac0d58d7ea69bc6a99ad1ce58aff39de15efee68a894596ad6581`.
The original remains Re4140/step525000, SHA
`58add6d20401c5df8f0a0cda3cfc421037e40e3da264df530d24af2f7cdc9c73`.
This short advanced state verifies restart machinery, not stationary relaxation.

`cavity_train.py` reuses the public PPO/DDPG/TD3 learner factories with the
author's 100-action horizon, .075 actuator limit and 34-probe preprocessing.
Initial states require a checked restart hash/header manifest; targets can be
the public example or an explicitly bound reconstructed calibration. The learner
now reads its actual configured horizon rather than hardcoding 200 for all cases.
Three synthetic 68-observation/one-action training/reload tests passed, including
rejection of changed targets, preprocessing and initial states. They are not CFD
performance evidence. `hg-cavity2d-eq36-restart-check-20260906` is testing two
actual 100-step native reset/replays before any full-budget cavity learner launch.

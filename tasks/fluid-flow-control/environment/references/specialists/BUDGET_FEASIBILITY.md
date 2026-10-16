# Runtime evidence and hour-scale development proposal

2026-09-07 02:17 UTC snapshot. User subsequently requested a reasonable setup
closer to one or two hours instead of a 15-day training run. No new long runs or
seed extensions were launched in this turn. Existing jobs were not terminated.
The budget scope (per track/seed, whole shared-model training, or entire five-case
reproduction including repeats) needs to be distinguished before choosing a new
release. No current data release or benchmark membership has been changed.

Follow-up: [single-GPU concurrency diagnostic](SINGLE_GPU_CONCURRENCY_20260907.md)
measured 1.53x aggregate throughput with four MPS environments on 2D Pinball.
This is not yet a measured 3D or end-to-end training speedup.

## Measured original-scale costs

Training and native CFD episode timings, not estimates from checkpoint size.
Full-run projections include scheduled evaluations and final/best replays;
small optimizer/reset overhead and changing throughput require a range.

| Case | Training controls / seed | Observed seconds / control | Approx. full run / seed on one H100 | Three seeds, accelerator-hours |
| --- | ---: | ---: | --- | --- |
| Pinball 2D Re100 / DDPG | 100,000 | .24–.27 | 7–8 hours | 21–24 |
| SquareCylinder 2D Re200 / PPO | 600,000 | .24 | 40–45 hours | 120–135 |
| Cylinder 2D Re1000 / PPO | 400,000 | .57 | 63–70 hours | 189–210 |
| Cylinder 2D Re3900 / PPO | 500,000 | .89–.91 | 120–135 hours | 360–405 |
| Pinball 3D Re150 / PPO | 100,000 | 12.61 | 15–17 days | 1,080–1,224 |

The three-seed total for these five method/case pairs is roughly 1,800–2,000
H100-hours per recipe round, excluding earlier tuning, extra algorithms and
debugging. Dividing total work by eight gives an ideal utilization lower bound,
not an attainable completion promise. Without within-run acceleration, the
critical path still contains a 15-day 3D seed even when three seeds run in parallel.
Training-budget completion does not guarantee paper-performance reproduction.

Source runs: `hg-pinball2d-ddpg-bootstrap-a025-s42-20260906` (completed in
24395.35s before final replay), its seed43 continuation; existing
`hg-cylinder2d-ppo-bootstrap-s42-20260906`; new Re1000 and Square full seed42
runs; `hg-pinball3d-ppo-bootstrap-check-20260906`. Use their `updates.jsonl`
elapsed-time slopes and `episodes.jsonl` physical timing. Re1000/Square remain
early estimates, 3D is extrapolated from only 400 training controls.

## CPU is supported, but did not accelerate this 3D case

Comparable 20-control fixed-action diagnostics at the same Pinball 3D Re150
physical configuration gave about 111.89 seconds/control on 8 CPU solver ranks,
versus 10.83 seconds/control on one H100 (mean per-step timing after the first
record, two episodes each). This is a measured approximately 10.3x GPU advantage
for that diagnostic, not a comparison against all host CPU cores or all solvers.
The first GPU episode also had substantial startup/warmup cost.

Evidence roots:
`hg-pinball3d-cpu8-fixed20-repeat-20260906/run` and
`hg-pinball3d-gpu7-fixed20-repeat1-20260906/run`.
Small CPU environments may still be useful for parallel rollout collection.
Moving this existing high-resolution case to CPU is not an hour-scale solution.

## Proposal, not a silently adopted benchmark change

- First exploit implementation improvements without altering physical inputs.
  A bounded 80M-vs-20M allocation-capacity diagnostic completed on GPU6 using
  the identical 16,465,722-cell supplied grid. Mean time after the first control
  was 10.7026 versus 10.6323 seconds (only 1.007x). The strict trajectory check
  also failed: maximum drag discrepancy was 2.37e-4, above the diagnostic 1e-6
  threshold. This does not isolate allocation as the cause, given previously
  observed 3D restart variation; it is insufficient to adopt the override.
  No long-training successor or changed reference score was produced. Original
  allocation settings remain in use. See `queue-stage1-capacity-20260907.sh`
  and its two preserved run directories in reproduction/.
- Eight-GPU parallel rollouts can improve wall time but have unmeasured scaling;
  exposing more GPUs alone does not parallelize this collector. Changing the
  number of environments or rollout/update schedule is a documented learner
  setup change, not automatically the original paper recipe.
- For a genuine 1–2-hour development budget, favor cheap 2D environments,
  validated lower-resolution training grids and an explicit interaction budget.
  Shortening interactions alone is not evidence of a useful learned baseline.
  Any coarse mesh must retain relevant shedding/control physics; score it
  independently and measure transfer to the chosen validation fidelity.
- High-resolution 3D may remain a small final validation component, not the
  per-update training environment. Existing cost is roughly 40 minutes for a
  single 200-control 3D rollout, so evaluation episode/seed counts also matter.
- Faster native alternatives within HydroGym, such as its JAX 2D Kolmogorov
  backend, are candidates rather than drop-in replacements for Pinball. The
  previous local 5,000-control JAX PPO run took 8,120.08s (~2.26h), establishing
  runtime feasibility only: its deterministic policy was no better than zero
  control and exact paper configuration remains unresolved. Do not relabel
  this existing diagnostic as a successful small-budget reproduction.

Primary documentation: [JAX backend](https://dynamicslab.github.io/hydrogym/docs/installation/jax/),
[configurable Kolmogorov example](https://dynamicslab.github.io/hydrogym/docs/examples/jax/kolmogorov/).

Fifteen days to two hours requires about 180x reduction. Even hypothetical
perfect eight-way speedup leaves about 45 hours for the unchanged workload.
Hour-scale AutoResearch and faithful original-scale baseline reproduction are
different deliverables. The next design should state which cost is bounded,
preserve the original results and label any new cheaper setup as a new version.

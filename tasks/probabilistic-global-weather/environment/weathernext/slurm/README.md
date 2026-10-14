# Optional WN2 cluster tools

The delivered track uses the [standalone commands](../docs/RUNNING.md) and
[starter](../environment/baseline/starter/README.md). Slurm is optional. Review the
partition, account and GPU directives before submitting any script; these
operator examples use site-specific defaults.

## Build and prepare

- `wn2_build.sbatch`: render the current Dockerfiles, build the starter and
  verifier Apptainer images, and run evaluator tests.
- [Resource preparation](../environment/data/MATERIALIZE.md): prepare the complete
  training, development and evaluation resources before allocating training GPUs.
- `prepare_wn2_cyclones.py`: generate the training cyclone labels; called by
  the canonical materializer.
- [WN2 operator reference](../wn2/README.md): profiling and curriculum tools.

## Train and evaluate

The main starter fixes 580 updates, including four updates at each AR3–AR8
stage, with a 24-hour training budget on four RTX PRO 6000 GPUs and 30 minutes
for packaging. Operator profiling scripts are diagnostics, not prerequisites
for running that starter.

[Unified submission evaluation](wn2_acceptance.md) documents the eight-date,
56-member, 15-day verifier and its separate GPU forecast and CPU scoring phases.
`wn2_harbor_gpu_boundary.sbatch` checks the inference namespace. Research
comparisons against GenCast and simple references live under
[paper-comparison](../paper-comparison/README.md).

## Implementation diagnostics

The remaining `check_wn2_*`, memory, attention, continuation and performance
probes support investigation of WN2 implementations. They may use smaller test
fixtures or different hardware; their outputs are not the track's main results.
See the [final evaluation report](../docs/validation/results-summary/README.md) and
[delivery validation](../docs/READINESS.md) for the reference result and validation scope.

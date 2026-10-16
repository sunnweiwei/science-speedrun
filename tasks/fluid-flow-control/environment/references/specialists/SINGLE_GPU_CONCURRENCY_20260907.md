# Single-GPU CFD concurrency: measured diagnostic

2026-09-07, completed at approximately 02:42 UTC. Manual runtime diagnosis only;
not an Agent run, learner change, new benchmark release or reproduced paper score.

Follow-up: [8/16-environment scaling with bracketing repeats](SINGLE_GPU_SCALING_20260907.md)
found no additional throughput benefit at eight and a regression at sixteen
for this same 2D case under default MPS settings.

## Results

One H100 (GPU6), unchanged pinned Pinball 2D Re100 physics, author-paper coupling
(170 CFD substeps per control), zero actions. Two 100-control diagnostic episodes
per worker: first warms the runtime, second measures throughput. Workers start
the measured episode together after reset. Startup is excluded from the main
metric and retained separately in raw results. One short trial per configuration.

| Configuration on one H100 | Total controls/s | Relative to ordinary one worker | Seconds/control for each worker | Exact measured-trajectory equality |
| --- | ---: | ---: | ---: | --- |
| Ordinary, 1 environment | 5.1805 | 1.000x | .1930 | Reference |
| Ordinary, 2 environments | 2.0858 | .403x | .9428–.9589 | Both pass |
| MPS, 1 environment | 5.0707 | .979x | .1972 | Pass |
| MPS, 2 environments | 7.0512 | 1.361x | .2769–.2836 | Both pass |
| MPS, 4 environments | 7.9238 | 1.530x | .4897–.5048 | All four pass |

The ordinary-four-worker branch was pruned during its first episode after
approximately 2.3 s/control/worker was observed. Partial logs are preserved, but
this branch is not included as a completed timing or correctness result.

Aggregate throughput is total measured controls / (last finish - first start),
not a sum of rates from non-overlapping windows. Measured start skew is at most
1.37 ms. All eight recorded numerical fields are compared at every measured
control: reward, drag, lift, observations, next observations, force coefficients,
normalized actions and physical controls. Every completed comparison matches
exactly numerically. MPS client registration was verified for 1, 2 and 4 solver
processes; merely starting a daemon was not counted as successful MPS use.

## Interpretation and limits

- Four MPS environments produced approximately 53% more transitions per second
  than one ordinary environment. Two MPS environments achieved approximately
  36% more. Single-environment latency does not improve; aggregate throughput
  does. Four environments provided only about 12% more throughput than two.
- Ordinary multiprocess execution became slower despite sampled GPU busy time
  increasing from about 65% to 95–99%. GPU busy time is not a measure of useful
  simulation throughput or fraction of peak compute. Context scheduling and
  synchronization are plausible contributors; no Nsight trace was collected
  here to attribute their individual costs.
- This result is for short zero-action 2D CFD trajectories, not end-to-end RL
  training, the other cases, or the expensive 3D Pinball environment. It does
  not establish a 53% training/convergence speedup, much less a 15-day-to-2-hour
  reduction. Further repeats and case-specific measurements are needed.
- The current learner advances a single environment serially. Exposing more
  GPUs or enabling MPS alone does not accelerate that loop. Concurrent rollout
  collection would need explicit integration. Multiple environments can feed
  one learner and one checkpoint; they do not require task-specific models.
- For original single-task reproduction, concurrency across independent seeds
  or evaluation episodes is a separate opportunity that need not alter the
  learning algorithm. Parallel environments within one training run change the
  sampling/update protocol and must be documented and validated accordingly.

No long-running training containers, published resources, host compute modes,
power settings or clocks were changed. Only the new ordinary diagnostic
container was stopped when pruning its slow branch. The MPS diagnostic completed
normally and shut down its private MPS daemon.

## Evidence and replay code

Full root: `/mnt/workspace/hydrogym/reproduction/concurrency-diagnostic-20260907`.

- `benchmark.py`: native-adapter wrapper, reset barriers, timing and comparison.
- `mps_only.py`: continuation of MPS comparisons after ordinary-four pruning.
- `source/`: copied baseline adapter and comparison code; large data is read-only.
- `pinball2d/summary.json`: complete machine-readable results and fieldwise checks.
- `pinball2d/<configuration>/worker-*/`: protocols, trajectories, timing and logs.
- Containers: `hg-single-gpu-concurrency-20260907` and `hg-single-gpu-mps-20260907`.
- Image ID: `sha256:1eb3775b0b4997b2b9fee6a68a8810f687879978a1fab7f7976d53f62266842d`.
- Dataset: `hydrogym-public-500gb-v1`, upstream data revision
  `cf5ef6748443ae72e9647f398487e98d421a7243`.

Primary references: [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
describes cross-process kernel concurrency;
[nvidia-smi utilization definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)
define GPU utilization as the fraction of time at least one kernel runs, not SM
occupancy or achieved FLOPS.

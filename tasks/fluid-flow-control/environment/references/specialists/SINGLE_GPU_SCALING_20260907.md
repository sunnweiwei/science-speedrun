# Does increasing MPS concurrency beyond four help?

2026-09-07, completed approximately 05:54 UTC. Bounded manual CFD diagnostic;
no training successor, new score, dataset change or production-learner change.

## Result

For the tested 2D Pinball case and default MPS resource settings, eight workers
did not improve aggregate throughput over four; sixteen made it worse. This is
a plateau of the current implementation/configuration, not evidence that the
H100's peak arithmetic throughput or memory bandwidth has been reached.

All runs used the otherwise idle host H100 GPU6. Source adapter, image, mesh,
control coupling and reward were unchanged. Each worker ran two 50-control
zero-action episodes: the first for warmup, the second for measurement following
a common reset barrier. This shorter timing horizon bounds the cost of high
concurrency. Use the contemporaneous controls below, not the previous
100-control timing windows as if they were the same experiment.

| Configuration, in execution order | Aggregate controls/s | Per-worker seconds/control | Exact measured-trajectory match |
| --- | ---: | ---: | --- |
| Ordinary 1, before | 5.4093 | .1849 | Reference |
| MPS 4, before | 7.9184 | .4840–.5052 | All four |
| MPS 8 | 7.7743 | .9929–1.0290 | All eight |
| MPS 16 | 6.6744 | 2.3183–2.3972 | All sixteen |
| MPS 4, after | 7.7779 | .4860–.5143 | All four |
| Ordinary 1, after | 5.2873 | .1891 | Pass |

Four-worker repeat variation was approximately 1.8%; ordinary-single repeat
variation was approximately 2.3%. Eight workers were essentially equal to the
lower four-worker repeat, so no meaningful benefit was observed. Sixteen workers
were approximately 14–16% slower than the four-worker controls. These repeats
measure drift, not confidence intervals; eight and sixteen had one trial each.

Aggregate throughput = all workers' measured control count / (last finish -
first start), not a sum of independent non-overlapping rates. Start skew was
under 2 ms. All eight numerical fields (actions, physical controls, observations,
next observations, rewards, drag, lift and force coefficients) matched exactly
at all 50 measured controls for every worker. MPS registration was verified for
the complete simultaneous sets of 4, 8 and 16 solver clients.

At sixteen workers, a sampled GPU memory footprint was about 15,121 MiB, well
below this card's capacity, and sampled GPU busy time was 100%. Spare memory
capacity therefore did not imply useful concurrency headroom. No Nsight trace
was collected here; compute, memory, scheduling and synchronization costs have
not been individually attributed.

## Recommendation and scope

Four environments per GPU is the economical starting configuration among those
tested for this 2D case. More environments consumed more resources and increased
individual latency without improving total throughput. This does not establish
the optimum for tuned MPS thread quotas, different CPU placement, another solver
implementation, other flow cases, or the much larger 3D environments.

This is zero-action CFD throughput, not measured learning/convergence speedup.
Concurrent rollout collection has not been integrated into the existing serial
learner. Multiple environments could feed a single model/checkpoint after that
integration, but neither the original learning schedule nor paper reproduction
can be assumed unchanged automatically.

The diagnostic completed with exit code 0; its private MPS daemon was stopped
normally and GPU6 was released. Existing training containers, host GPU settings
and the frozen benchmark release were not modified.

## Evidence

Root: `/mnt/workspace/hydrogym/reproduction/concurrency-scale-diagnostic-20260907`.

- `scale.py`, `benchmark.py`, copied `source/`: finite plan and native wrapper.
- `pinball2d/plan.json`, `pinball2d/summary.json`: plan and complete comparisons.
- Per-worker protocol, trajectory, timing and console logs, plus MPS logs.
- Container: `hg-single-gpu-mps-scale-20260907`.
- Image: `ai4sci-hydrogym-baselines:20260905`, ID
  `sha256:1eb3775b0b4997b2b9fee6a68a8810f687879978a1fab7f7976d53f62266842d`.
- Data: `hydrogym-public-500gb-v1`, upstream revision
  `cf5ef6748443ae72e9647f398487e98d421a7243`.
- Native adapter SHA256:
  `3c81bbfa599f47c4e8949522b66435a08acbce073ae85d48f8dd7a1c48261aa9`,
  identical to the previous concurrency diagnostic's adapter.

[NVIDIA MPS documentation](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
describes both cross-process concurrency and optional resource provisioning;
the latter was not tuned in this experiment.

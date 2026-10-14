# WN2 native AR8 memory investigation

This document preserves successive memory experiments. Measurements apply only
to the implementation, hardware and batch named in each section. Current
training/readiness is tracked in [READINESS.md](READINESS.md); do not interpret
an earlier failure or peak below as the status of the current model.

The historical pre-packed-mask/pre-cuDNN implementation had a sampled device peak of 96,934 MiB
(94.66 GiB). This is not a theoretical minimum. The first successful update
was job 12533424; diagnostic 12535457 reproduced the sampled peak and produced
a final XLA buffer assignment. Job 12535457 completed both native updates in 27m23s; its final weights hash
exactly matches the earlier independent two-update run 12533550. Its second
update took 392.28 seconds including checkpoint I/O.

## Compiler allocation accounting

The `accumulate` executable has the following disjoint allocation categories,
whose sizes sum to 94.55 GiB:

| Category | GiB | Meaning |
| --- | ---: | --- |
| Parameter allocations | 5.22 | Executable arguments, including weather inputs/targets and gradient accumulator; not just model weights |
| Constants | 8.11 | Mostly attention mask construction constants |
| Output allocations | 0.685 | Predominantly the returned accumulated parameter gradient |
| Temporary allocation | 80.53 | Shared workspace containing buffers with different lifetimes |

The optimizer's Adam state lives outside this gradient executable. These
compiler categories must not be added to a separate model-state estimate or
treated as an exact process-wide physical-memory ledger. Executable constants,
aliases, runtime allocations and actual buffer residency require separate
accounting.

At the compiler's reported peak, the five largest values are each 4.456 GiB.
Their operation metadata identifies mesh-to-grid edge MLP/recomputation and
gather-adjoint operations, with 3,114,720 edges and 768 BF16 channels. Six other
large values have shape `f32[8,2,1,2,13,721,1440]`, each 1.609 GiB, in the
autoregressive loop's saved state. Peak-value sizes are not additional
allocations to add to the table.

## Attention is another substantial allocation source

The released configuration selects Splash attention. The current GPU adapter
uses native three-block attention with the same 32-hop mask. CPU construction
of the actual permuted graph gives 40,962 nodes, block width 10,273 and four
blocks. Three logit blocks across six heads contain 7,598,486,088 elements:
14.15 GiB at BF16 and 28.31 GiB at FP32. Native softmax upcasts its operands.
These logical sizes do not establish their simultaneous physical residency.

An additional source-level issue is that native three-block mask padding uses
int32 and promotes boolean adjacency masks. XLA allocates 7.863 GiB of
`s32[10273,10273]` constants. A boolean-mask prototype preserves mask truth
values/layout in CPU tests and reduces logical mask storage fourfold. Six GPU FP32/BF16 attention tests subsequently passed exact forward and
parameter/input-gradient equality (job 12536975). Native HRES AR1 subsequently
passed two real updates and exact fresh-process optimizer continuation on one
PRO6000 (12539905), with a sampled peak of 60,996 MiB. This does not verify AR8
memory or four-device continuation.

The first native HRES AR1 boolean-mask attempt (12538281) compiled its gradient
but failed before executing it: its executable serialization was 2,380,774,846
bytes, exceeding protobuf's 2 GB limit. Boolean storage alone therefore does
not make persistent executable caching possible at native resolution. The AR1
compiler allocation was 56.49 GiB, including 2.19 GiB of constants; this is a
different autoregressive horizon from the AR8 table and is not an end-to-end
memory-saving comparison. The successful replacement disabled executable
caching and retained saved autotuning.

Bit-packing masks on the host and unpacking behind an optimization barrier
preserves exact values and all tested FP32/BF16 GPU gradients. Native HRES AR1
job 12540130 then passed two updates and exact fresh-process continuation; its
weights hash also exactly matches boolean-mask job 12539905. Gradient compiler
constants decreased from 2.19 to 0.467 GiB, and total allocations from 56.49 to
54.77 GiB. Serialization succeeded, producing a 116,790,953-byte compressed
cache entry. This compressed size is not an uncompressed executable estimate.

The resumed process still missed the gradient cache: cold dump and warm
load/require-complete autotuning flags participate in the pinned JAX cache key.
AR8 job 12540247 added a third process with identical warm flags. Its original
Slurm exit was a false negative in a case-sensitive cache-log regex. An
independent CPU recheck of all three saved checkpoints and the actual log
passed: all state tensors match exactly and the third gradient executable
hits the cache. No GPU rerun or numeric tolerance change was involved.

Across that full AR8 run, the sampled device peak was 84,550 MiB (82.57 GiB),
versus 96,934 MiB (94.66 GiB) for the original masks. Compiler allocation fell
from 94.55 to 78.64 GiB; constants were 0.494 GiB and temporary allocation
72.24 GiB. The final weights SHA-256 matches both original AR8 runs exactly.
The validated packed-mask function is now included directly in `native_model.py`
and production profiles enable executable caching. The canonical package passed
one-date, one-lead independent development inference in the v4 verifier image
(job 12540683). Updated images, final ten-day candidate replay and full-budget
four-device profiles still require validation.

The experimental query-chunk implementation retains all native keys and mask
edges. Its CPU FP32 value/adjoint oracle passes. BF16 forward outputs match in
the tested synthetic cases, but gradients differ by approximately 0.4–0.6%
relative L2. It is not enabled in production, and small synthetic tests do not
establish full-model equivalence or runtime improvement.

A separate cuDNN BF16 attention experiment preserves projections and the three
neighbor blocks, concatenates their keys, and pads odd sequence lengths for
cuDNN backward. Three small GPU cases repeat exactly in process, but differ
from native BF16 forward results by 0.37–0.45% relative L2. The largest small
case improves measured forward/backward time by only about 13%. It is not
enabled in production. At coarse native shape (two blocks of width 5153, six
heads, width 768), synthetic forward/backward time improved from 117.8 to
54.1 ms, and compiler temporary memory from 5.43 to 2.33 GiB. Forward relative
L2 difference was 0.42%; the synthetic mask and benchmark do not establish
full-model quality or replay. Full-native AR1 job 12540125 failed before its
first update with `raw_hash_map::at` during gradient compilation/preparation.
A two-layer regression (12540759) reproduced this failure with the original
backend configuration. The pinned XLA compiler sets `force_deterministic` only
on the first backward graph cache miss; repeated equivalent calls retain a
different fingerprint. Setting that flag in the JAX backward configuration
before XLA hashing passed all three two-layer GPU cases. The deterministic
GPU policy and pinned JAX 0.6.2 remain required.

With this isolated fix, native AR1 job 12540971 completed two updates and two
fresh-process continuations in 24m30s. Every checkpoint tensor matched exactly,
and the third process explicitly hit the gradient executable cache. The second
initial update took 29.61s including input preparation and checkpoint writing;
gradient dispatch/execution took 12.01s. Compiler allocation was 33.30 GiB and
the sampled GPU peak was 38,140 MiB (37.25 GiB). Cached fresh-process timing
includes executable loading and must not be interpreted as steady kernel time.
These are one-device, batch-one AR1 diagnostics, not four-device training-budget
measurements.

AR8 job 12541022 subsequently passed all three processes with zero differing
state leaves and an explicit third-process gradient cache hit (39m36s total).
Its steady update took 150.16s, including 37.54s input preparation, 108.91s
gradient execution and 3.39s checkpoint writing. Compiler allocation was
57.35 GiB; the initial measurement sampled 62,760 MiB (61.29 GiB). The final
memory report is retained in the linked continuation evidence. Independent
v4 verifier job 12541280 also passed one development case/lead: weather reward
0.7345089898 and precipitation reward 0.6221781721. These scores do not establish
full ten-day quality.

The validated attention and deterministic configuration functions are now
embedded in `native_model.py`; an AST comparison confirmed equivalence to the
tested diagnostic functions, and the pinned-image import/idempotent-hook check
passed. Canonical four-PRO6000 batch64 stage0 profile 12543876 passed at 48.88s
per steady update, with 656.20s setup/compilation/export overhead. Production-policy
three-process replay 12543877 subsequently failed continuous-versus-resumed state.
Changed BF16 arithmetic still requires final trained candidate evaluation; the
full curriculum, budget, four-device exact continuation and final ten-day replay
remain open gates.

Full ten-day cuDNN inference replay (12542582) also passed for eight members
and all 40 leads, with exact decoded bits and schema for all 18 stored arrays.
Cold/warm forecasts took 845.28s/827.03s; the job including comparison took
56m59s. Compressed bytes differed, as allowed by the decoded-value contract.
This used the two-update AR1 diagnostic package; final trained weights still
require their own complete replay and held-out scores.

## Multi-device transfer correctness

Profile 12542414 failed when the new checkpoint guard indexed a remote GPU
through JAX. Copying each shard to the host before indexing fixed that access
error. A separate two-A5000 audit then reproduced a more fundamental problem:
`device_put_replicated([1, 0])` produced `[0, 0]` on the second device, while
independent host-to-device copies preserved both replicas. Test 12543869 passed
with the replacement helper, including host argument distribution, `pmean`,
and rejection of bitwise parameter or optimizer replica differences. This is
evidence of an observed transfer-path failure, not an established upstream or
driver root cause. The canonical trainer now distributes parameters, optimizer
state and random keys from host buffers. Full-model within-process replica equality and stage0 timing then passed in
12543876. However, 12543877 differed in 1455 state leaves between continuous
and resumed training, while its two restored processes matched exactly and
hit the same gradient cache. Update2 loss matched across all three processes,
but gradient norm was 46.18846 continuous versus 43.23490 restored. Gradient
and optimizer compilation cache keys matched across all three processes.
Job 12543883 records each device's gradients before collective averaging to
locate the divergence; this does not yet establish a collective or backward
root cause. No full-curriculum budget or continuation acceptance is claimed.

## Evidence

- [Four-PRO stage0 profile](validation/wn2-host-replication-four-pro-profile.json)
- [Four-PRO continuation failure and cache-key audit](validation/wn2-host-replication-four-pro-continuation-failure.json)
- [Two-GPU host transfer and replica regression](validation/wn2-host-replication-gpu.json)
- [XLA allocation and peak-operation metadata](validation/wn2-ar8-xla-memory.json)
- [Native graph and attention shapes](validation/wn2-native-attention-shapes.json)
- [Boolean-mask CPU checks](validation/wn2-boolean-mask-cpu.json)
- [Native boolean-mask cache failure](validation/wn2-boolean-mask-native-cache-failure.json)
- [Native boolean-mask continuation](validation/wn2-boolean-mask-native-continuation.json)
- [Native packed-mask continuation and cache limitation](validation/wn2-packed-mask-native-continuation.json)
- [AR8 exact continuation, cache hit and final memory](validation/wn2-packed-mask-ar8-continuation.json)
- [cuDNN GPU measurements](validation/wn2-cudnn-attention-gpu.json)
- [cuDNN coarse-shape measurements](validation/wn2-cudnn-native-shape-gpu.json)
- [cuDNN native failure](validation/wn2-cudnn-native-failure.json)
- [Repeated-layer deterministic cache regression](validation/wn2-cudnn-cache-regression.json)
- [Fixed cuDNN native AR1 continuation and memory](validation/wn2-cudnn-fixed-native-continuation.json)
- [Fixed cuDNN native AR8 continuation and memory](validation/wn2-cudnn-fixed-ar8-continuation.json)
- [Fixed cuDNN independent development package smoke](validation/wn2-cudnn-fixed-package-smoke.json)
- [Full ten-day cuDNN decoded exact replay](validation/wn2-cudnn-full-ten-day-replay.json)
- [Canonical packed-mask development package smoke](validation/wn2-canonical-packed-package-smoke.json)
- [Query-chunk CPU checks with native mask dimensions](validation/wn2-query-chunks-native-mask-cpu.json)

Remaining work is to validate the canonical inference package, four-device
continuation and the full training budget, and any candidate optimization without
relaxing the full curriculum or independent evaluation requirements.

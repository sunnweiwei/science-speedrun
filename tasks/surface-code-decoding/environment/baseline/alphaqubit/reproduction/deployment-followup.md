# Deployment follow-up (2026-09-25)

This change follows the from-scratch protocol introduced in PR #51 and the
reference launcher fix in PR #64. It does not change datasets, model code,
training updates, inference numerics, submission schemas, score anchors or
historical measured results. The independent H100 scratch training reported in
PR #65 remains in progress; this change does not claim its final accuracy.

## Conditions for the resource change

Task v1.2 raises the agent allocation from 2 H100 GPUs for 24 hours to **8 H100
GPUs for 48 hours**, while preserving random initialization and the absence of
agent-visible trained weights. The verifier still receives one GPU and its
existing 3,900-second total timeout.

The estimate applies to the unchanged `solution/solve.sh` reference recipe:

- Ten specialists, one visible GPU per specialist, scheduled in whole waves.
- Each has an 18-hour pretraining cap and up to two hours of fine-tuning
  (fine-tuning can stop earlier at its step limit).
- Eight concurrently available GPUs yield two waves, approximately 36–40 hours
  plus packaging/validation overhead. Four GPUs imply three waves (54–60 hours),
  and two GPUs imply five waves (90–100 hours).
- The estimate assumes the existing microbatch/runtime fits the allocated host
  and device memory. It is not an H100/H200 throughput conversion or an accuracy
  guarantee. Faster pretraining executes more steps within the time cap.
- The allocation leaves roughly 8–12 hours beyond one reference run, not a budget
  for multiple complete retraining campaigns.

The original measured costs are 179.9625 GPU-hours of pretraining and
19.9466667 GPU-hours of fine-tuning on RTX PRO 6000 Blackwell Max-Q 96 GB cards;
see `alphaqubit-summary.json`. Dividing their sum by eight misses the launcher's
second partial wave. Queue delays in the historical 47.9-hour wall time are not
part of this recipe estimate.

## Deployment and precision checks

Development networking stays runner-owned: task TOML defaults to no network;
an operator may select a model API allowlist through Harbor. Hardcoding Compose
`network_mode: none` would prevent that allowlist sidecar from attaching. Verifier Compose
requests exactly one NVIDIA GPU; local replay inherits it. The trusted grader
requests none. Native Docker CI resolves that topology before using an explicit
CPU-only override for the synthetic isolation fixture. CI builds both real
images and starts the development image offline using the hash-verified public
HF development archive. Fixture scores validate infrastructure, not model quality.

Two tests that compare mathematical FP32 results now scope
`jax.default_matmul_precision("highest")` to the compared operations: attention
against its NumPy formula and microbatch gradient accumulation against a full
batch. Their tolerances remain unchanged; production precision is untouched.
The reviewer's two H100 failures did not include exact test IDs/logs. On our
A5000, the original attention test failed under the unset default precision
(maximum absolute difference 0.00044505), and passed with the local precision
scope. An explicit ambient `tensorfloat32` setting also passed the original
attention test on this device, so it is not equivalent to the unset default.
No H100 validation was performed here.

The exact interrupted-versus-continuous training test is explicitly CPU-only.
GPU investigation found matching parameters, EMA, both RNG states, sampler
state and counters, but optimizer moments differed by at most 9.32e-10 in the
default-precision run. `highest` precision still did not provide bitwise GPU
optimizer equality. An independent save/load roundtrip preserved every value,
shape, dtype and weak-type flag; GPU kernel/reduction nondeterminism is plausible
but its precise cause is not established. The CPU exact-equality assertions and
portable serialization checks are preserved, with no global tolerance changes.
Run this resume contract with `JAX_PLATFORMS=cpu`; GPU runs explicitly skip it.
This does not establish bitwise reproducible GPU training.

CPU checks passed: 32 release-contract tests and all 11 model/training tests.
Native Docker and final GPU evidence is recorded in the PR's validation section.
Public HF development/verifier roles were freshly downloaded to a new cache and
materialized with the current pinned revisions. No data or trained weights were
uploaded or changed by this follow-up.

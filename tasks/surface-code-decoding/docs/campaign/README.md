# CS AlphaQubit reproduction

The primary target is **AlphaQubit (Bausch et al., Nature 2024)** on the Sycamore
d3/d5 memory experiments. The implementation uses JAX/Haiku, the published
recurrent Transformer core, and the Sycamore model dimensions. It is an independent
reconstruction from primary sources, not official Google source or weights.

The user explicitly selected AlphaQubit after reviewing the earlier QRENND
baseline choice. QRENND training was interrupted and its remaining jobs cancelled.
Sixteen valid best/last checkpoints from eight trained groups are preserved.
The prior campaign consumed **145.90194 actual GPU-hours**; this cost remains
inside the conservative 384-GPU-hour total. No H200-equivalence is inferred.

See [plan.md](plan.md) for the current contract and resource gates,
[alphaqubit-commands.md](alphaqubit-commands.md) for executable commands,
[alphaqubit-source-audit.md](alphaqubit-source-audit.md) for the exact source
checklist and unspecified details, and
[alphaqubit-implementation-options.md](alphaqubit-implementation-options.md)
for the independent audit of community implementations and GPU options.
The user-accepted v1 reference and original RSI design proposal are recorded in
[benchmark-spec.md](benchmark-spec.md). The released [agent instructions](../instruction.md)
and [runtime contract](../environment/contract.md) now define materialized HF
roles, generic model delivery, separate Harbor verification, and a normalized
score anchored by the strongest same-test released tensor-network reference.
The compact [reference summary](alphaqubit-summary.json) includes exact split
counts, model sizes, measured results, compute and source-evidence hashes.

## Current evidence

- The first full AlphaQubit run completed on 2026-09-22 at 02:16 UTC. All ten
  groups finished both stages, followed by frozen-package replay and scoring on
  3.25M held-out odd shots per stage. Independently audited fine-tuned LER is
  **3.01265% at d3 and 3.22889% at d5**. This beats standard/correlated MWPM at
  both distances; d3 slightly beats released TN, while d5 remains worse than
  TN and belief matching. The Nature reference is 2.901%/2.748%, under its
  different fold/ensemble protocol. Official-level accuracy is not established.
- At v1 completion cumulative use was **346.33722 GPU-hours**, including earlier
  LSTM work. The supplementary d5 continuation completed at30k steps, used
  3.71139 further GPU-hours and reached3.21858% d5 test LER. Total allocation is
  now350.04861 GPU-hours, with no active campaign GPUs. The fixed v1 reference
  remains unchanged; the continuation is a post-test follow-up on the same fold.
- Official data archive verified: 130 surface-code conditions, 6.5M shots.
- Full standard and correlated matching replay completed on CPU; stored TN/BP
  predictions independently rescored. All paired shot counts remain available.
- AlphaQubit input adapter verified against all 130 real geometries and allowed
  even raw measurements/sweeps. Poisoned odd labels do not affect development.
- Model tests verify the supplement's attention formula, spatial pooling,
  causal prefixes, terminal masks, finite gradients, and parameter reload.
- BF16 computation retains float32 master parameters. Runtime pins and lockfile
  are separate from the retained TensorFlow/QRENND environment.
- Full-size FP32 and BF16 d3/d5 GPU probes completed on RTX PRO 6000 Blackwell
  Max-Q. At 25 rounds, batch256, BF16, measured rates were about 2442/1526
  samples per second; peak memory was 9.61/24.46 GB. These are steady-state
  optimizer probes, not streaming decoder latency.
- A fixed d3 three-round batch128 overfit diagnostic passed after 88 updates
  (2/128 main errors, main BCE 0.093509). This is a training sanity check only.
- Independent integrated checks:96 passed in the JAX environment (one legacy
  TensorFlow test skipped); all33 prediction-contract tests passed in the retained
  TensorFlow environment. Fresh relocated inference reproduced66,560 synthetic
  validation probabilities exactly after deleting the original model directory.
- Representative full-size training and its source-bound learning gate passed;
  the full training/fine-tuning/replay/scoring dependency chain then completed.
  Its original allocation ceiling was359.036944 GPU-hours including prior work,
  and it never exceeded eight concurrent campaign GPUs.
- Additional gate/report checks:17 passed. Independent CPU sampling/MWPM replay
  on both full validation populations passed; this is not a trained model score.

The completed first result is a bounded single-seed, even-train/odd-test
reproduction. Convergence and parity with the Nature headline results are not
claimed; the original paper also reports ensembles and both cross-validation
directions. Full measurements and the continuing d5 gap are recorded in plan.md.

## Data and protocol

Use even-fitted pij DEMs to stream simulation examples with the published
noise curriculum. Every batch has one uniformly sampled experiment duration
from 1,3,…,25 rounds. Model inputs contain measurements reconstructed by cumulative
XOR of detectors, events, and their published temporal/boundary context.
Official logical-path flips are targets; no raw logical label is an input.

Hardware adaptation uses the first 19,880 even shots per condition for training,
the remaining 5,120 even shots for validation, and 25,000 odd shots for test.
The released even-fitted DEMs include validation syndromes but no validation
logical labels. Pretraining and fine-tuning checkpoints are selected on
validation, frozen, and then scored on the held-out half. Public held-out data
are not an intrinsically private test set.

The source audit distinguishes published settings from necessary reconstruction
choices (including residual-network internals, value-head width, cycle embedding
and ambiguous schedule counters). The model does not replace the recurrent core
with a generic Transformer or GRU. Larger-distance Pauli+ and soft-readout results
are outside this first Sycamore campaign.

## Runtime and artifacts

Install the alphaqubit-runtime directory using uv sync --frozen --python 3.11
and invoke alphaqubit-runtime/.venv/bin/python from the task directory.
The GPU environment uses JAX 0.7.2, Haiku 0.0.15 and pinned CUDA12 dependencies.
Set XLA_PYTHON_CLIENT_PREALLOCATE=false; let the pinned packages supply CUDA libs.

Persistent data, checkpoints, reports, and logs stay outside Git at
/home/users/xy200/yinxunjian/mycode/ai4sci-artifacts/surface-code-decoding/20260919.
The campaign ledger retains every original submission cap. Accounting combines
settled actual use for retired jobs with hard caps for ongoing work, so an
interrupted campaign cannot hide spent compute or block reclaimed unused budget.

The retained [commands.md](commands.md) and [upstream-audit.md](upstream-audit.md)
describe the earlier QRENND campaign; do not rerun its cancelled full chain as
the AlphaQubit procedure. This work is not yet a completed Harbor task.

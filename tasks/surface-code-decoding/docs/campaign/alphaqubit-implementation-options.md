# AlphaQubit implementation and GPU options audit

Audit snapshot: 2026-09-20 01:31 UTC. This is a source and environment review,
not an AlphaQubit training result or a measured throughput comparison. No GPU
allocation was created, no existing job was changed, and no community source
was copied into the task implementation.

## Recommendation

Implement the published Sycamore AlphaQubit configuration independently from
the primary paper and supplementary pseudocode, preferably with JAX/Haiku and
a compiled recurrent scan. Neither inspected community repository is a
trustworthy licensed drop-in reproduction. A single RTX PRO 6000 pilot is the
best available candidate at the last inventory snapshot; a typed A5000 request
is the practical fallback. Actual compile time, examples/second, memory use,
and learning must be measured before assigning the remaining campaign budget.

The original implementation used JAX, Haiku, and JAXline. A scan is an
implementation choice that should preserve the per-round computation; it does
not justify changing the architecture, objective, optimizer, or data protocol.
Use one model and one GPU first. Independent seeds or condition groups can
later use separate GPUs without introducing distributed-training changes.

## Primary-source contract

Sources are the [Nature paper](https://www.nature.com/articles/s41586-024-08148-8),
Methods, Extended Data Figures 4 and 8, and the [official supplementary
information](https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41586-024-08148-8/MediaObjects/41586_2024_8148_MOESM1_ESM.pdf).
Local copies are in `ASSET/references/alphaqubit/`. The supplementary PDF's
SHA-256 is `139c7b4317437e7f87f554ff729eac7053a248c1580cfd2a88f9f286444a9982`;
the community repository's mirror has the same hash.

The directly specified Sycamore defaults are:

| Component | Published requirement |
| --- | --- |
| Input | Stabilizer measurement and detection-event features; learned stabilizer index; two residual layers; distinct terminal input representation and undefined off-basis embedding |
| Recurrent state | One 320-dimensional vector per stabilizer; combine old state and new embedding as `(state + embedding) / sqrt(2)` |
| Syndrome Transformer | Three layers per round; pre-normalized attention and gated-dense residuals; four heads, key dimension 32 |
| Gated dense | First projection 320 to 1600; GELU on first 800 entries multiplied by the other 800; project 800 to 320 |
| Convolution | Three residual 3×3 convolutions per layer, 160 channels, 1×1 return projection where needed; stabilizers scattered to the physical `(d+1)×(d+1)` grid |
| Dilations | d3: `(1,1,1)`; d5: `(1,1,2)` |
| Attention bias | For pij: static pair-feature embedding dimension 48, eight residual layers, plus seven dynamic event-correlation indicator features; separate head projections in each Transformer layer |
| Readout | 2×2 spatial convolution to data-qubit representations, dimensionality reduction, pooling perpendicular to the chosen logical lines, residual readout with 16 layers and width 64; one designated line used at inference |
| Optimizer | LAMB for Sycamore; beta2 0.95; pretraining weight decay 1e-5 on non-bias parameters; finetuning decay 0.08 toward pretrained parameters |
| Auxiliary loss | Next-stabilizer cross-entropy, weight 0.02 |
| Learning rate | d3: 3.46e-4; d5: 2.45e-4; warm-up 10,000 steps; published piecewise reductions by 0.7 |
| Batch / averaging | Batch 256 then 1024; EMA constant 0.0001 |

Attention bias is optional across the paper's experiments, but is specified
for the pij configuration. Its omission in Pauli+ experiments is not evidence
that it should be removed from a pij Sycamore reproduction. The paper's
Sycamore input is hard readout; its later soft-readout Pauli+ experiments are
a distinct training and data regime.

Several details remain implementation assumptions rather than fully specified
source facts: beta1, optimizer epsilon and precise LAMB trust-ratio convention;
EMA initialization/debiasing; internal feature/readout ResNet operations;
some projection dimensions and cycle-number embedding details. Extended Data
Figure 4 includes a cycle-number embedding. The supplement only specifies
attention, gated-dense, convolution, and recurrent core pseudocode. Community
choices do not resolve these gaps.

There are also notation inconsistencies: the self-attention pseudocode refers
to undefined `Z` for K/V; one convolution projection refers to an undefined
intermediate; several dimensions omit head concatenation. Shape-consistent
self-attention is the clear intended operation. Bias is inside the division
by `sqrt(key_dimension)` in Algorithm 1. Document any interpretation explicitly.
The noise-curriculum text refers to training steps, while Figure 8 labels the
transition in examples; batch-change/schedule units also need an explicit,
source-linked interpretation. Do not substitute a community scheduler silently.

DEM pretraining samples lengths uniformly from `{1,3,...,25}`, uses the
training-half fitted DEM, a discrete noise-scale curriculum, and independent
synthetic validation. Hardware finetuning uses the first 19,880 examples from
that same half, with the remaining 5,120 for selection. Odd/even are reversed
for the published two-fold result. The bounded campaign may use one fold and
one seed, but must report that difference. Pretraining with the circuit/XEB
DEM or finite cached examples must be identified as such.

## Community source inspection

Both public repositories were cloned into `/tmp/alphaqubit-community-audit-*`
for inspection. Neither tracked a LICENSE file or provided a source-header
license grant at the inspected commit. Public visibility alone is not a
reusable code license. The recommendation is independent implementation from
the published methods, with these repositories used only as comparison points.

### overshiki/alphaqubit-reproduce

Inspected commit: `9acd2a232a44d8cb62cfd02123e9ca37d5f4fe8d`, branch `rep`.
[Source tree](https://github.com/overshiki/alphaqubit-reproduce/tree/9acd2a232a44d8cb62cfd02123e9ca37d5f4fe8d).

The source in `alphaqubit/model.py` closely follows the *core* supplement:
scaled recurrent addition, pre-normalization, GELU gated dense, spatial
scatter, residual dilated convolutions. These facts are useful cross-checks,
but the rest of the pipeline has material differences:

| Source | Concrete finding and implication |
| --- | --- |
| `alphaqubit/alphaqubit.py` | Enabling attention bias raises `NotImplementedError`; no static pair embedding or seven indicator features are wired into the forward pass. Not the published pij architecture. |
| `alphaqubit/alphaqubit.py`, `readout.py` | Cycle embedding exists only as an option and defaults off. Readout uses a direct 320-to-64 2×2 convolution, omitting the separately described dimensionality-reduction stage. Default d5 dilation is `(1,1,1)` unless a caller overrides it. |
| `alphaqubit/readout.py` | Its own orientation comments identify X as horizontal and Z as vertical, then pool columns for X and rows for Z: along those lines. This differs from the paper's perpendicular pooling; the physical-coordinate and observable convention must be derived independently. |
| `alphaqubit/inputs.py` | Final embedder has an extra on-basis projection and no stabilizer-index embedding; feature/readout residual internals are an assumed pre-norm MLP. These choices are not established by the supplement. |
| `alphaqubit/training.py` | Uses AdamW with PyTorch defaults, including beta2 0.999 and epsilon 1e-8, instead of Sycamore LAMB beta2 0.95. Finetuning regularization toward pretrained parameters is not implemented by this optimizer helper. |
| `alphaqubit/training.py` | Noise curriculum samples a continuous uniform interval whose lower bound follows a sigmoid. This is different from the paper's six discrete DEM scale factors with Gaussian-mixture weights. The piecewise scheduler describes itself as a simplified stand-in. |
| `scripts/train_pipeline.py` | Default warm-up 100, gradient clipping 1.0, auxiliary weight 0.01 and EMA decay 0.9999. EMA uses the conventional previous-shadow coefficient; this alone does not establish original initialization. |
| `alphaqubit/stim_data.py` | Both `si1000` and `pauli_plus` dispatch to the same approximate Stim-generated CSS circuit. There is no gate-level Pauli+ leakage/cross-talk simulator here. These noise names cannot support a Pauli+ reproduction claim. |
| `alphaqubit/stim_data.py` | Detector cumulative XOR is used to construct measurements, fixing omitted first-round off-basis information. Optional synthetic I/Q is then applied to stabilizer inputs while the original hard event arrays remain unchanged. This is not the published soft-event construction. |
| `scripts/train_pipeline.py` | Every output line is trained against the same single Stim observable label. No distinct per-line labels are supplied. This is not the published per-line supervision for simulations with all logical-observable labels available. |

No weights, verified Sycamore fold result, or independently verified
Nature-matching learning curve was obtained from this inspection. README
claims about reproduction, scale, or throughput are not acceptance evidence.

### rmalcewicz/AQ1

Inspected commit: `3194afce35c4220b40e744e8f504b6bfdea031ea`, branch `main`.
[Source tree](https://github.com/rmalcewicz/AQ1/tree/3194afce35c4220b40e744e8f504b6bfdea031ea).

`model.py` uses width 128, four standard Transformer-like blocks, dropout,
one-dimensional convolution over stabilizer index, a learned scalar bias
matrix shared between heads, a GRUCell after the Transformer stack, and mean
pooling of all stabilizers into a short MLP. The position embedding is fixed
to eight stabilizers. It has no physical 2D residual convolution stack,
published gated-dense module, scaled recurrent combination, or perpendicular
readout. It is a different d3 decoder, not a faithful AlphaQubit baseline.

`generate_data_full.py` generates Stim **detector samples**.
`train.py` reshapes these directly into groups of eight, treats them as raw
measurements, and XORs adjacent rows again to make its events. This loses the
correct first/final on-basis detector layout and double-differences detector
data. `train.py` uses Adam and validation BCE selection. Neither this pipeline
nor the repository's IQM claims establish a Sycamore reproduction.

## CS runtime and inventory evidence

Read-only `sinfo` / `scontrol show node` were used. Availability changed during
the audit, so these are snapshots, not reservations. At 01:30:52 UTC:

| Node / type | Configured / allocated GPU count | Practical implication |
| --- | --- | --- |
| fitz-47, RTX PRO 6000 | 4 / 2 | Two GPUs unallocated at snapshot; best candidate for the first pilot |
| fitz-45,46,48,49, RTX PRO 6000 | 4 / 4 each | Fully allocated at final snapshot |
| A5000 nodes | Approximately 25 GPUs unallocated in preceding aggregate snapshot | Broad fallback availability; request typed A5000 gres |
| gpu-compute6, V100 | 3 / 0, IDLE | Available fallback requiring a Volta-compatible runtime |

RTX PRO 6000 nodes have no feature tag. Use the configured typed request
`--gres=gpu:rtx_pro_6000:1`, not a nonexistent Blackwell constraint. Mixed
physical-GPU nodes also require typed gres. CPU and memory availability,
reservation/QOS rules, and the eventual scheduler decision still apply.

Existing environments were inspected without modifying them. Imports and
compiled CUDA architecture flags were checked on the current CPU node:

| Existing environment | Verified PyTorch / CUDA | Compiled architectures |
| --- | --- | --- |
| `/usr/project/xtmp/xy200/mycode/_massspecgym_repro/venv_rtx` | 2.7.1+cu128 / 12.8 | sm75,80,86,90,100,120 and compute120; suitable candidate for Blackwell/A5000, **not V100** |
| `/usr/project/xtmp/xy200/mycode/_massspecgym_repro/venv` | 2.7.0+cu118 / 11.8 | Includes sm70 and sm86; suitable candidate for V100/A5000, **not Blackwell** |

These are useful compatibility references, not permission to mutate another
project's environment or a GPU execution test. Create a task-specific pinned
environment. PyTorch's [2.7 release notes](https://pytorch.org/blog/pytorch-2-7/)
confirm Blackwell support through its CUDA 12.8 builds.

No JAX GPU environment was found in the bounded project `.venv` and conda
environment searches. A fresh Python 3.11 environment with pinned JAX CUDA 12,
Haiku, and Optax is the broadest candidate if the same implementation must also
run on V100. The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
states CUDA 12 supports SM5.2+ and requires Linux driver 525+, whereas CUDA 13
requires SM7.5+ and driver 580+, excluding V100. For Blackwell, the actual
compiler/toolkit must support sm120 (CUDA 12.8+); inspect installed `ptxas` and
the full forward/backward kernel path rather than relying only on package
names. Avoid an inherited `LD_LIBRARY_PATH` overriding wheel CUDA libraries.

The current CPU node has no `nvidia-smi`. A read-only SSH query to fitz-49
failed host-key verification; no host-key bypass or GPU allocation was used.
Consequently **GPU driver version and real GPU execution remain unverified**
until the pilot's runtime check.

## Bounded pilot acceptance evidence

Within the already declared eight GPU-hour diagnostic allowance:

1. Before training, record GPU name, driver, compute capability, JAX and CUDA
   package versions. Require a GPU device explicitly; fail on CPU fallback.
2. Compile the exact d5 architecture for representative short and 25-round
   batches. Record compile time separately from synchronized steady-state
   forward/backward time. First use float32; compare mixed-precision finite
   losses/gradients before enabling it for long runs.
3. Confirm single-step and scan outputs/gradients agree on a small deterministic
   fixture. Check that final readout branches do not update the recurrent state
   used by a continuing trajectory, and that terminal inputs contain only
   allowed stabilizer parities.
4. Overfit a small training-only sample, then require improvement on an
   independently seeded synthetic validation set. Report BCE and per-round
   LER fit quality; do not select from odd-test performance.
5. Measure examples/second and observed peak memory at the intended batch and
   round distribution. Project the planned examples and adaptation allowance
   against approximately 238 remaining allocated GPU-hours. This is the point
   at which a full campaign duration becomes defensible.

No pilot result, guaranteed reproduction accuracy, or H200-equivalent runtime
is inferred from GPU names or community README timing claims.

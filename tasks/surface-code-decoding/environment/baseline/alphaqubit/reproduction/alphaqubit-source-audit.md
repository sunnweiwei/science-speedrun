# AlphaQubit Nature 2024: primary-source implementation audit

Audited 2026-09-19/20 against the published article, Methods, Extended Data
Figures 4/5/8, and official Supplementary Information. This is a reconstruction
checklist for the **Sycamore distance-3/5, pij-pretrained variant**. It does not
describe a generic Transformer, AlphaQubit 2, or the earlier QRENND LSTM.

The paper specifies the recurrent core well, but does **not** fully specify an
executable architecture or every optimizer/data convention. Unresolved choices
below must remain explicit configuration/provenance fields. A new implementation
can reproduce the documented design; parameter-level identity with the authors'
unreleased implementation cannot currently be certified.

## Primary sources and retained files

- [Published article and Methods](https://www.nature.com/articles/s41586-024-08148-8),
  DOI `10.1038/s41586-024-08148-8`, Nature 635, 834–840 (2024).
- [Published PDF](https://www.nature.com/articles/s41586-024-08148-8.pdf), 28 PDF pages.
- [Official Supplementary Information](https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41586-024-08148-8/MediaObjects/41586_2024_8148_MOESM1_ESM.pdf),
  four PDF pages: cover followed by printed pages 1–3 / Algorithms 1–5.
- [PMC article mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11602728/).

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `nature-2024.pdf` | 5,849,106 | `565794ee61a72c7d62a6e094dbd60ed826ddced241bcb7c7d810ae18cc830e28` |
| `supplementary-information.pdf` | 590,990 | `139c7b4317437e7f87f554ff729eac7053a248c1580cfd2a88f9f286444a9982` |

Both PDFs have corresponding `.txt` files extracted using `pdftotext -layout`.
`extended-data-4.png` and `extended-data-8.png` are rendered from published PDF
pages 22 and 26. The hyperparameter table is an image: the values below were
read from that rendering, not inferred from incomplete text extraction.

Page references below mean **one-based PDF pages** unless explicitly described
as supplementary printed pages. Most architecture details are main pp12–13;
training is pp14–15; input-frame conventions are p10.

## Scope and input contract

Let `d` be the code distance, `S=d²−1` the number of stabilizers, and `D=320`
the Sycamore representation width. Thus `S=8` for d3 and `S=24` for d5.
The paper uses the XZZX circuit variant, with X/Z type labels inherited from
the Clifford-equivalent CSS layout. “On-basis” refers to stabilizers inferable
from preparation/final data measurements, not simply the literal X/Z gate
letters of the XZZX circuit. [Methods, pp8 and12.]

- [x] Preserve one vector per stabilizer throughout the recurrent state,
  shape `[batch,S,D]`, initialized to zero.
- [x] Supply both **measurement** and **detection-event** features for the
  Sycamore binary-input model. The optional third/fourth inputs are leakage
  probability and its temporal-difference analogue, for the leakage experiments.
- [x] For binary inputs, `event[i,t] = measurement[i,t] XOR measurement[i,t−1]`.
  Measurements can be recovered by cumulative XOR of events in a fixed initial
  frame. First off-basis events are zeroed; for Pauli noise the initial random
  off-basis stabilizers are XORed into later values so that this frame is zero.
  This is explicit in “Soft measurement inputs versus soft event inputs,” p10.
- [x] For `R` physical ancilla-measurement rounds, use `R` bulk frames and one
  final frame inferred from measured data qubits. Figure4b shows `S0,...,S_N`,
  with the final frame handled specially. Each provided frame invokes the same
  recurrent core. Do not drop the final boundary or insert an extra physical
  round when translating this indexing convention.
- [x] The final frame contains only `(d²−1)/2` on-basis stabilizers. Its on-basis
  values use **separate final-round projection parameters**. Undefined final
  off-basis inputs use one learned embedding. Normal first-bulk off-basis
  zeros do not imply using this final-only undefined embedding. [p12.]
- [x] Targets are the logical-observable **error/flip**, using the observed
  final logical path and initial logical state. They are not QRENND's all-data
  parity target. Only logical error labels are outputs; raw logical/data-parity
  labels must never enter model features.
- [ ] For soft-readout extensions, threshold final **data-qubit** readouts
  before forming final stabilizers, including leakage readouts. The paper
  identifies soft final stabilizers as a possible label-information leak.
  Sycamore pij reproduction uses binary inputs and does not require I/Q data.

The six checked hard-input requirements are implemented in
`alphaqubit_model.py`/`alphaqubit_data.py` and covered by model/data tests,
including all130 official geometries and allowed even hardware measurements.
The unchecked soft-readout extension remains outside the current scope.

The source supports the gauged cumulative-XOR representation for the Pauli/DEM
variant. It does not justify substituting that representation for raw soft
measurement information under state-dependent amplitude damping.

## Architecture checklist

### Stabilizer feature embedding

For each stabilizer, sum learned linear projections of its input features and
an index embedding, then apply a two-layer residual network to obtain the
`D`-dimensional stabilizer representation. The index embedding distinguishes
physical stabilizers. The feature projections are shared across positions;
the position/index embedding is learned per stabilizer. Final features have
the special treatment above. [Methods p12; Fig4c; Fig8b.]

The supplement does not define this residual network's internal activation,
number of linears per residual unit, normalization placement, or initializer.
The table's “2 layers” must not silently become two arbitrary large MLPs
without recording the interpretation.

### Recurrence and syndrome transformer

**Algorithm5, supplementary PDF p4 / printed p3** defines:

```text
x = (previous_state + embedded_current_stabilizers) / sqrt(2)
repeat for three distinct syndrome-transformer layers:
    x = x + multihead_attention(layer_norm(x), attention_bias)
    x = x + gated_dense_block(layer_norm(x))
    x = scattering_residual_convolution_block(x)
```

Core parameters are reused across time; the three layers within a core have
their own parameters. This is additive scaled recurrence, not LSTM/GRU gating
or the later AlphaQubit2 recurrent design. The figure rounds the scale to0.7;
Methods says0.707; Algorithm5 gives the precise `1/sqrt(2)` expression.

### Multihead attention

There are four heads, each with query/key dimension32. Each head has biased
linear Q/K/V projections and the concatenated result has a biased output
projection. The learned attention bias has a separate projection per layer and
head. [Algorithms1–2, supplementary PDFp2 / printedp1; Fig8b.]

With column-oriented token notation, Algorithm1 specifies:

```text
Q = Wq X + bq
K = Wk X + bk
V = Wv X + bv
score = K^T Q + bias
output = V softmax(score / sqrt(32))
```

In row-oriented implementations this is the corresponding query–key transpose
and softmax over keys. Crucially, the literal pseudocode divides **both QK and
bias** by `sqrt(32)`. An implementation that adds bias after scaling QK is a
different convention unless its bias projection is rescaled accordingly.

Algorithm1 writes undefined `Z` for the K/V inputs despite declaring only `X`;
the self-attention interpretation requires `Z=X`. The value-head dimension is
**not resolved by the publication**: Algorithm2's head-concatenation and output
projection shapes omit the required head factor. Choices such as value width80
(`D/4`) and32 (equal to key width) are distinct reconstruction assumptions.

### Gated dense block

**Algorithm3, supplementary PDFp3 / printedp2** fixes widening `w=5`:

```text
u = Linear(D, w*D)(x)            # 320 -> 1600
left, right = split(u, 2)        # two 800-channel branches
y = Linear(w*D/2, D)(GELU(left) * right)  # 800 -> 320
```

Both linears have biases. This is not two separate 1600-channel branches,
nor a sigmoid gate. The prose describes modulation “between zero and one,”
but Algorithm3 explicitly uses GELU; the algorithm is the operational source.
The surrounding residual addition and preceding LayerNorm are in Algorithm5.

### Scattering residual convolutions

**Algorithm4, supplementary PDFp3 / printedp2**:

1. Scatter the `S` stabilizer vectors into their `(d+1)×(d+1)` spatial grid.
   Empty positions receive an optional learned padding vector `P`.
2. For each of three convolutional residual updates, normalize the current
   grid over channels, apply a dilated3×3 convolution to160 channels and GELU,
   then project with a1×1 convolution back to320 channels and add to the grid.
3. Gather the actual stabilizer locations back to `[batch,S,D]`.

All three Sycamore convolutions use160 output channels before their1×1
projections. Dilations within each transformer layer are `(1,1,1)` at d3 and
`(1,1,2)` at d5. [Fig8b,c.] Spatial shape must remain constant through each
residual update; the exact edge-padding rule is not stated. Learned padding
for *empty internal lattice sites* should not be confused with the numerical
padding used outside the grid by a dilated convolution.

Algorithm4 line9 references undefined `X-tilde`; it must use the just-computed
convolution output `Y-tilde` to be dimensionally meaningful. The code should
record this repair to the published pseudocode.

### Attention bias: enabled for Sycamore pij

The bias pathway is present for the **pij** Sycamore variant. Figure8b marks
its48-dimensional embedding, eight residual layers and seven indicator
features “only for pij.” The XEB/SI1000 and Pauli+ configurations should not
automatically inherit it. [Methods p13; Fig8b.]

Construct a learned `[S,S,48]` embedding by summing discrete feature embeddings
for each stabilizer pair `(i,j)`:

- coordinates of both stabilizers;
- signed coordinate offsets;
- Manhattan distance;
- whether their inherited stabilizer types match.

Process this through the eight-layer residual network. This geometric tensor
is independent of the current state and can be cached at inference. The
embedding is shared input to separate layer/head bias projections.

At each frame concatenate the following four pairwise event products:

```text
e_t[i]*e_t[j], e_t[i]*e_(t-1)[j],
e_(t-1)[i]*e_t[j], e_(t-1)[i]*e_(t-1)[j]
```

and the three distinct diagonal features of these matrices. There are seven
indicator channels total, producing55 channels before the per-layer/head
projection. The paper does not spell out whether the three diagonal vectors
are re-embedded as diagonal matrices or broadcast over a pair axis. A literal
diagonal-matrix interpretation is reasonable, but must be declared. Previous
events at the first frame are naturally zero; this initialization is not
separately formalized in the algorithms.

### Spatial readout and auxiliary prediction

After the final core update, scatter to the grid and apply a2×2 convolution
to create one representation per **data qubit**, combining its four adjacent
stabilizer sites. Then reduce dimensionality and mean-pool along rows/columns
**perpendicular to the logical-observable lines**. Apply a residual network
and a binary output to obtain one prediction per equivalent logical line.
For Sycamore, the readout residual network has16 layers, width64, widening1.
[Methods p13; Fig4f; Fig8b.]

Only the conventional leftmost/lowest logical observable is used at inference,
depending on basis. If all equivalent-line labels are available, their losses
can be averaged; this is described for the scaling simulation, not required
for the released single-observable Sycamore DEM samples. Pooling along the
logical line because it appears geometrically natural contradicts the explicit
perpendicular-pooling statement in the paper.

Figure4f also supplies an embedding of **cycle number** to the readout pathway.
Its formula, feature scaling, dimensionality, and exact addition/concatenation
point are absent from the text and supplement. Omitting it or choosing
`Linear(log1p(cycle))` must be documented as a reconstruction decision. A
finite learned lookup table alone needs an explicit extrapolation rule if
claiming arbitrary-duration decoding.

An auxiliary head linearly projects each stabilizer state to a logistic
prediction of the **next stabilizer**. Its BCE is averaged across rounds and
stabilizers and weighted0.02 relative to the logical-error objective for
Sycamore. The primary text does not fully specify next-measurement versus
next-event targets or inclusion/masking of final inferred stabilizers.

## Published Sycamore optimization and data protocol

| Item | Published value / behavior | Source |
| --- | --- | --- |
| Frameworks | JAX, Haiku, JAXline | p14 |
| Optimizer | LAMB, not Adam/AdamW and not the scaling-model Lion optimizer | p14, Fig8b |
| Beta2 | **0.95** | Fig8b image |
| Pretraining weight decay | `1e-5`, non-bias parameters, toward zero | p14, Fig8b |
| Fine-tuning weight decay | `0.08`, non-bias parameters, toward pretrained parameters | p14, Fig8b |
| d3 peak initial LR | `3.46e-4`, pretraining and fine-tuning | Fig8c |
| d5 peak initial LR | `2.45e-4`, pretraining and fine-tuning | Fig8c |
| d3/d5 LR relation | approximately `sqrt(2)`, not2 | Fig8c |
| LR warm-up | linear, first10,000 training steps | p14 |
| LR reductions | multiply by0.7 at `{0.8,2,4,10,20}×10^5` steps | Fig8b |
| Pretraining batch size |256 then1024 | p14, Fig8b |
| Batch-size change | table says “step” `4×10^6`; the counter is not defined more precisely | Fig8b |
| Fine-tuning batch size |1024 | p14 |
| Parameter EMA constant |0.0001; averaging direction/formula not supplied | Fig8b |
| Next-stabilizer BCE weight |0.02 | Fig8b |
| pij/XEB pretraining termination |up to2billion examples | pp14–15 |
| SI1000 pretraining termination |500million examples | p15 |
| Fine-tuning maximum |30,000 training steps | p14 |
| Parameter selection |EMA weights with smallest development fitted LER, with fit-quality rejection | p15 |
| Ensemble |20 seeds for Sycamore; average **logits**, include all models | p15 |

The beta1, epsilon, precise LAMB trust-ratio/decay implementation, gradient
clipping, initializer, precision, dropout, and optimizer-state transfer between
phases are not supplied. “Use the default optimizer” is insufficient provenance
unless the exact implementation/version/defaults are also recorded. A
finetuning-only low learning rate such as the earlier LSTM's `1e-4` is not a
published AlphaQubit Sycamore hyperparameter.

### pij pretraining and noise curriculum

Use the released DEM fitted to the same experimental fold that will later be
used for fine-tuning/validation. Sample lengths uniformly from
`{1,3,...,25}`. Pretraining validation is independently sampled from that same
fold's DEM; selection uses fitted logical error per round, not simply BCE.
The published training does not require reusing a fixed20M/60M sample cache.

At noise factors `f∈{0.5,0.6,...,1.0}`, scale each DEM error-mechanism
probability by `f`. Do not replace the calibrated DEM with a generic
depolarizing circuit and call it the pij variant. Draw the factor with
probability proportional to

```text
w(f,t) = 1 + w_c * exp(-(f-f_c(t))² / (2*sigma_c²))
f_c(t) = f_min + (1-f_min)/(1+exp(-s_c*(t/t_c-1)))
```

Then normalize across the six factor choices. Figure8a gives
`w_c=12`, `sigma_c=0.05`, `f_min=0`, `t_c=12,800,000 training examples`,
`s_c=1`. **Do not silently replace `f_min=0` with0.5** even though the factor
grid begins at0.5. Equations6–7 call `t` training steps, whereas the table labels
`t_c` in examples. Using examples-seen for this curriculum is a defensible
explicit resolution, not an unambiguous optimizer-update schedule.

The table's separate batch-size change counter also needs an explicit policy.
Interpreting `4e6` literally as optimizer updates is possible within a2billion
example run (the first4M updates at batch256 consume1.024billion examples).
Interpreting it as4M examples changes training materially. The source does not
justify silently switching units for convenience. Record all three counters:
optimizer updates, examples seen, and the active learning-rate/noise schedules.

### Fine-tuning and evaluation splits

The paper performs two-fold cross-validation: train on even-indexed examples
and evaluate on odd, and then swap. Within the development fold, each of13
round settings has25,000 examples: first19,880 for fine-tuning and final5,120
for validation. This is258,440 training examples and66,560 validation examples
per patch/basis/fold. The opposite25,000 examples per setting are held out.
At batch1024,30,000 updates correspond to about119 passes over the fine-tuning
training population, consistent with the reported roughly120 passes.

The data-derived DEM itself was fit using the whole development fold, including
the eventual validation subset; this is the publication's protocol. It does
not authorize use of the opposite evaluation fold's labels for fitting,
selection, schedule changes, or model comparisons during development.

For LER selection, fit

```text
F(n) = 1 - 2*error_rate(n)
log F(n) = intercept + n*log(1-2*LER)
```

using rounds3,5,...,25, excluding r1 boundary effects. Reject parameter sets if
`R² <= 0.9` or fitted intercept `<= max(-0.02,-sigma_intercept)`. The authors
retain EMA parameters giving the lowest acceptable validation LER. The
evaluation frequency and exact validation sample count for synthetic
pretraining are not specified. Neither a poor early fit nor NaN should be
turned into an artificially attractive zero LER.

## Parameter counts and unresolved details

**No exact total d3/d5 parameter count is stated in the primary article or
supplement for the Sycamore320-channel configuration.** The main text reports
approximately **5.4million parameters for the scaling model** across code
distances (PDFp4); that uses256 channels and different readout settings in
Figure8. The source says counts remain nearly constant across distances apart
from stabilizer-index embeddings. Community claims of approximately8.3M for
Sycamore are not a source-verified acceptance target.

The gated-dense block alone is fully countable: each Sycamore block has
`320*1600 + 1600 + 800*320 + 320 = 769,920` parameters. Under biased QKV/output
linears, a value width32 gives164,544 attention parameters per layer; value
width80 gives287,616. The difference is369,216 parameters across three layers.
Therefore the unresolved value size prevents an authoritative total count
even before accounting for unspecified residual/readout blocks. Record the
implementation's exact per-module count after all choices are fixed.

Required reconstruction decision ledger:

1. Value-head dimension and attention-bias scaling convention.
2. Repair of undefined symbols and inconsistent dimensions in Algorithms1/2/4.
3. LayerNorm axis, epsilon, learned scale/offset, numerical precision.
4. GELU exact/approximate variant and all initialization choices.
5. Embedder/bias/readout residual-unit internals and parameter sharing.
6. Empty-site learned-padding sharing and external convolution padding.
7. Readout2×2-convolution channels/activation, subsequent projection ordering,
   cycle-number embedding, and final logical-line indexing.
8. Coordinate-feature lookup sharing and diagonal-indicator expansion.
9. Auxiliary target/mask convention, especially final off-basis values.
10. LAMB beta1/epsilon/trust-ratio details; anchored finetuning decay;
    optimizer-state reset or transfer.
11. EMA update formula and initialization; warm-up/reduction counters;
    batch transition counter; curriculum counter; restart behavior.
12. Synthetic validation population/seeds, checkpoint-selection frequency,
    multi-device effective batch semantics, and bounded-compute deviations.

The currently proposed independent reconstruction chooses value width32,
pre-LN two-linear GELU residual units, and a `log1p(cycle)` scalar projection.
The root's preliminary count for the scaling configuration is approximately
5.39M with value width32 versus5.58M with value width64. Agreement of the former
with the paper's rounded5.4M supports, but does not prove, the width32 choice:
other under-specified modules also affect the total. This audit has not claimed
an exact official Sycamore parameter count from that indirect comparison.
These are **implementation assumptions**, not facts recovered from the paper.
The source supports reporting a single-model, single-fold, compute-bounded
AlphaQubit-design reproduction if clearly labeled; it does not support calling
that the full20-seed/two-fold published result.

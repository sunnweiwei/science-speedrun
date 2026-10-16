# QRENND source and protocol audit

Audit date: 2026-09-19. This reproduction uses the native QRENND LSTM model
implementation with a task-specific data adapter and bounded training protocol.
Preserving the model architecture does not imply reproducing the published
training protocol or logical error rates exactly.

## Source and environment

The complete package is vendored unchanged under `baselines/reference/qrennd/`
from [Hossein-D/qrennd](https://github.com/Hossein-D/qrennd) commit
`c2ed73e3ec6ee843293c8839bf059c63b2960e61`, a mirror identifying Boris M.
Varbanov as author and `BorisVarbanov/qrennd` as its original repository. The
original address was unavailable during the audit. Apache-2.0 licensing is
preserved in `baselines/reference/LICENSE`, and each copied file has a SHA-256
hash in `baselines/reference/provenance.json`.

The wrapper imports and invokes native `Config` and `get_model`; there is no
TensorFlow-to-PyTorch port or rewritten model. Original dependency metadata pins
Python 3.8.12, TensorFlow 2.4.1, and NumPy 1.19.5. The reproduction uses Python
3.11, TensorFlow 2.15.1 / Keras 2, and NumPy 1.26.4. Native imports also require
xarray, scipy, networkx, matplotlib, and PyYAML.

The [TensorFlow compatibility table](https://www.tensorflow.org/install/source)
documents CUDA 12.2 / cuDNN 8.9 for TensorFlow 2.15. A5000 is the initial target.
Use a verified modern runtime before moving to a Blackwell sm120 GPU. Model
construction and CPU checks do not establish GPU throughput; profile an actual
training step before setting a compute schedule.

## Architecture and objective

Source line references below refer to unchanged files under
`baselines/reference/qrennd/`.

- `models/networks.py:11–38`: two ordinary Keras LSTMs. The first returns a full
  sequence; the second returns the final hidden state. No internal LSTM dropout
  is enabled. Keras defaults supply tanh state activation, sigmoid gates,
  Glorot input initialization, orthogonal recurrent initialization, and unit
  forget bias. An external 5% dropout follows only the second LSTM.
- `models/models.py:18–60`: a ReLU follows the recurrent output. The main head
  concatenates the terminal defect vector with this output. The auxiliary head
  sees recurrent output alone.
- `models/networks.py:123–160`: each head is Dense(H, ReLU), external 5%
  dropout, Dense(1, sigmoid). No L2 penalty is configured.
- `models/models.py:62–71`: native Adam compilation; configured learning rate
  `0.0005`. Both heads receive the same binary target. Loss is
  `BCE(main, target) + 0.5 * BCE(aux, target)`; accuracy is diagnostic. Scoring
  uses only the main output.

The [Keras LSTM documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
describes the default activations and accelerated cuDNN execution conditions.
The external dropout does not enable recurrent dropout or change those defaults.

| Distance | Recurrent width F | Terminal width E | Hidden width H | Parameters |
| --- | ---: | ---: | ---: | ---: |
| 3 | 8 | 4 | 64 | 60,418 |
| 5 | 24 | 12 | 253 | 926,488 |
| 5, historical config only | 24 | 12 | 128 | 244,738 |

For these two equal-width LSTMs and heads, the count is
`14 H² + 4 H F + H E + 12 H + 2`.

## Published settings versus historical configs

The experimental-noise rows of Table I prescribe d3 H64/batch64/20M shots and
d5 H253/batch256/60M shots, with 5% dropout and learning rate `5e-4`. **253 is
printed in both the arXiv and published article**; it is retained literally.
The paper describes patience20, best-validation-loss checkpoint selection, and
occasional additional training at a reduced learning rate. Its logical-error
fit excludes the one-round condition. Sources: [arXiv v2, Table I and Sec.
VI.B](https://arxiv.org/html/2307.03280v2), [published PRR 7, 013029, Table I and
Appendix A.2](https://doi.org/10.1103/PhysRevResearch.7.013029).

The retained `reference/scripts/configs/base_config_google_d{3,5}.yaml` files
instead configure 155,000 shots × 13 odd round counts × 10 states = 20.15M
training shots, batch64, patience10, and up to 500 epochs. Their separate dev
and test directories each contain 5,000 × 13 × 10 = 650,000 shots by config.
The d5 file uses H128. Thus these configs are historical examples, not the final
published d5 prescription. They are preserved as evidence, not used as the
training configuration.

Native `datasets/generators.py:15–29` loads fixed netCDF groups. Native
`datasets/sequences.py:31–105` materializes groups in memory and produces a
batch from one round/state group at a time. Keras shuffles the batch indices;
the sequence has no within-group epoch shuffle hook. A streaming simulator,
different epoch definition, smaller sample budget, or changed batch size is a
training-protocol adaptation and must be recorded in the run manifest.

## Input and observable adaptations

Native preprocessing (`datasets/preprocessing.py:26–54,262–292`) XORs the ideal
frame out of ancilla measurements, calculates time differences of the resulting
syndromes, and derives terminal defects by XORing the last matching-basis
syndrome with stabilizer parity projected from data-qubit flips. Native labels
are the parity of **all** data-qubit flips relative to ideal measurements.

The task uses the official Sycamore detection events and converts its observable
labels to the native global-data-parity target. Detection events are mapped by circuit spatial/time coordinates into
`(rounds, d²−1)` recurrent inputs and a separate `(d²−1)/2` terminal vector.
First-round opposite-basis detectors do not exist in the released circuit;
their recurrent entries are filled with zeros. The native measurement pipeline
can contain initial random opposite-basis stabilizer outcomes. These input
conventions differ even though they encode the same relevant detector history.

The official observable uses a weight-d logical string; native QRENND's all-data
parity differs from it by stabilizer parity. `baselines/observables.py` derives
that parity by solving a GF(2) row-space equation using only the ideal circuit's
measurement-record annotations. The resulting detector parity XORs the official
label into the native global-data-flip target for **both heads**. At inference,
the same parity converts the main probability back to the official path-flip
probability. The original training objective is therefore preserved rather than
replaced by a different logical representative. Each run records the exact
detector-index certificate and circuit hash for every condition. No target bit
is added to input features; raw hardware measurements and test labels are not
used to derive this transform.

The transform was checked independently against all-data measurement-flip
parities on hardware and synthetic shots across all 130 release conditions, as
well as generated Stim circuit fixtures. See `tests/test_observables.py` and the
transform module for the executable derivation and checks. The missing random
first-round opposite-basis features remain an input-convention difference, so
this is not an exact reproduction of the original measurement input pipeline.

The original study trains on simulation and evaluates experiment. The released
Sycamore data's even/odd split recommendation specifically avoids fitting a DEM
on the same hardware shots being decoded. Neural adaptation splits, if used,
must also keep validation/test labels separate from training and checkpoint
selection. They are a task protocol, not a QRENND package requirement.

## Simulator fidelity and limits

The original study describes an experimental circuit-level Pauli model taken
from Google's released experiment, with operation-dependent error rates. It
explicitly excludes leakage, crosstalk, and stray interactions. These omissions
also limit the released simulator used here; they are not additional physical
error channels discarded by our adapter. See [Sec. II.B and IV of the
paper](https://arxiv.org/html/2307.03280v2) and the [Sycamore data
release](https://zenodo.org/records/6804040), whose README describes
`circuit_noisy.stim` as the hardware circuit with an approximate Pauli model.

We sample the released `circuit_noisy.stim` files directly in pinned Stim 1.15,
rather than rebuilding circuits with the authors' separate circuit-construction
wrapper. Auditing the d3 and d5 X-basis 25-round pilot circuits found only
`DEPOLARIZE1`, `DEPOLARIZE2`, and `X_ERROR` noise instructions. The full circuit
files and cache payloads are hashed. This is source-grounded use of the same
documented noise-model family, not verified bit-for-bit regeneration of the
authors' original training population. Their raw simulation data and complete
training scripts were available by request, and we do not possess that exact
population or its original seeds.

Our cache uses the zero-sweep reference. State-independent Pauli channels and
Clifford propagation make detector/observable *flip* distributions invariant to
the initial computational bitstring after ideal-frame correction. As a local
check, the d3/d5 X-basis 25-round circuits produced identical 1,024-shot
detector/observable-flip arrays at seed 144 when their sweep-controlled initial
X gates were fixed to zero, all-one, or alternating patterns. This check does
not imply such invariance for physical amplitude damping or leakage, nor does
it restore the missing random initial opposite-basis features described above.

These experiments reproduce a native QRENND baseline using the released Pauli
simulator. They are not an AlphaQubit architecture port, an AlphaQubit Pauli+
simulation, or evidence of recovering either paper's published final scores.

## GPU pilot audit and training allocation

`reproduction/pilot-summary.json` records two 1,000-step native pilots on
separate Tesla V100-PCIE-32GB GPUs. Each artifact's source hashes, model config,
parameter count, Adam iteration, embedded cursor, validation cache hash, and
observable-transform certificates were checked independently. Full 66,560-shot
synthetic validation was replayed on CPU: both main/aux error counts matched
the GPU records exactly, and combined loss differed by less than `7e-9`.
Slurm reports 50 and 61 allocated seconds, totaling 0.03083 GPU-hours.

Batch-only steady throughput extrapolates to 1.093 hours per 20M-shot d3 epoch
and 0.957 hours per 60M-shot d5 epoch. One epoch across all eight d3 and two d5
groups would therefore consume about 10.66 GPU-hours before callback/startup
overhead. This extrapolates two X-basis pilots and is not a measurement of all
ten groups or other GPU types. The pilots cover only 0.32% and 0.43% of one
epoch; their near-chance error rates provide no convergence assessment.

The existing 240-GPU-hour primary allocation allows roughly 20–24 epochs per
group at these rates, with throughput and callback variability. Start with the
representative d3/d5 pair and examine full-epoch validation trends before
spending the remaining primary allocation across all groups. Preserve the
existing separate adaptation/replication, evaluation, and reserve allocations
inside the 384 actual allocated-GPU-hour ceiling; use measured Slurm accounting
to authorize continuations, rather than an invented H200 conversion.

The paper's stopping rule is 20 **epochs** without validation improvement, not
20 timed validation callbacks. It reports neither a total epoch count nor
training GPU-hours that would certify convergence inside our budget. Its CPU
inference timings and hypothetical hardware data-acquisition estimate are not
training-runtime estimates. The current bounded runner deliberately omits that
early-stopping rule and records nonconvergence until stronger evidence exists.
See [Sec. VI.B](https://arxiv.org/html/2307.03280v2) for those distinctions.

`baselines/train.py` applies the native combined BCE to fixed cached simulation
or the allowed even-index hardware shots. It shuffles shots within conditions
and homogeneous-round batches each epoch, evaluates full validation populations
at timed batch boundaries and epoch ends, and saves full `.keras` archives with
Adam state and an embedded data cursor. Resume preserves the epoch permutation
and skips completed batches, while explicitly recording that the dropout RNG
restarts. Changes to data adapter, native wrapper, or observable-transform source
hashes block resume; training-runner and cache-tool revisions are recorded per
session so operational fixes remain auditable. Bounded runs omit early stopping
and do not infer convergence from reaching a walltime, epoch, or diagnostic cap.

## Verification contract

`tests/test_neural.py` checks vendored hashes and both expected parameter
counts; runs a native CPU forward/backward step and checks finite metrics and an
actual parameter update; and compares both outputs before and after saving and
reloading weights. These checks establish executable model construction and
checkpoint parity. They do not establish learned decoding accuracy, convergence,
GPU efficiency, or paper-level reproduction. Those require measured training
and evaluation results in the run records.

Verified on the CS node using the task environment on 2026-09-19:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  .venv/bin/python -m pytest tests/test_neural.py -q
```

Result: **4 passed in 32.44 seconds**. Both model sizes completed an optimizer
step and passed checkpoint parity. The runtime emitted 32 Keras/Python
`randrange` deprecation warnings; there were no failed or skipped tests.

The training runner's CPU integration tests subsequently passed **4 tests in
20.28 seconds** using the same environment and thread settings:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  .venv/bin/python -m pytest tests/test_training.py -q
```

They exercise timed partial-epoch validation, native optimizer checkpoint
restoration across step limits 3 → 5 → 40, a resumed epoch boundary, exact sample
accounting, metric-state isolation during validation, semantic-source change
rejection, corrupted-cache rejection, nonempty-output protection, and allowed
hardware-label loading with deliberately invalid odd-index test labels. This
is execution/protocol validation on small simulated fixtures, not a benchmark
accuracy result. The combined native/trainer suite also passed eight tests
before the additional epoch-boundary assertions were added.

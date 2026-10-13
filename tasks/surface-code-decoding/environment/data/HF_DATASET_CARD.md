---
language:
- en
license: cc-by-4.0
pretty_name: Sycamore Surface-code Decoding — AI4Sci Materialized Release
size_categories:
- 1M<n<10M
task_categories:
- tabular-classification
tags:
- quantum-error-correction
- surface-code
- sycamore
- alphaqubit
- ai4science
---

# Sycamore surface-code decoding: materialized benchmark

Predict a logical observable flip from repeated stabilizer detection events in
a noisy quantum memory. The benchmark trains decoders that improve the
reliability of encoded quantum information. It uses real Sycamore hard-readout
experiments at code distances 3 and 5, not simulated soft-readout d11 data.

![Scientific problem, real validation example and output](science-introduction.png)

Source: [Google Quantum AI Sycamore memory experiments, Zenodo 6804040](https://zenodo.org/records/6804040),
CC-BY-4.0. Scientific model reference:
[Bausch et al., Learning high-accuracy error decoding for quantum processors,
Nature 635, 834–840 (2024)](https://www.nature.com/articles/s41586-024-08148-8).
The reference AlphaQubit implementation is an independent reproduction, not
official Google code or weights, and does not reproduce the paper's headline
ensemble accuracy. Its trained weights are not distributed to agents.

## Contents and splits

- `development.tar.gz`: physically separate train and validation arrays,
  ideal circuits, published even-fitted detector error models, and `reference/`:
  the reference model's validation predictions, validation metrics, aggregate test
  score and training cost, plus a submission template without weights.
- `verifier-inputs.tar.gz`: input-only test arrays and ideal circuits. No logical
  labels, raw measurement records, historical decoder predictions, or odd-fit
  noise models are in this archive.
- `release-manifest.json`: archive sizes, expanded sizes and SHA-256 identities.
- `NOTICE.md`: attribution and source/licensing notes.

Each of 130 conditions contains 50,000 original shots: four d3 patches and one
d5 patch, X/Z bases, and 13 odd round counts 1..25. Original zero-based rows are
split into first 19,880 even rows for train, last 5,120 even rows for validation,
and all 25,000 odd rows for test. Totals are 2,584,400 / 665,600 / 3,250,000.
There is no geometry or acquisition-session holdout. IDs are opaque uint64
values; the inference interface does not expose original source row numbers.

Training/validation NPZs contain `sample_ids:uint64[N]`, packed
`detectors:uint8[N,ceil(D/8)]` (little-endian bit order), and `labels:uint8[N]`.
Verifier inputs contain only the first two. A split manifest specifies geometry,
array paths, ideal circuits, shot counts and hashes. Targets are the official
logical-path observable flips, not raw final-data-qubit parity. Noise DEMs are
calibrated using all even syndromes, including validation syndromes, but no
logical labels or odd test data. This is a declared calibration exception.

The original dataset and historical test are public. A separate private
`Corning/ai4sci-surface-code-decoding-evaluation` repository supplies labels to
trusted benchmark operators; this runtime separation does not make the source
historically secret. Never expose test roles to an agent's development runtime.

## Model and score

The reference model has a recurrent Transformer core with width 320, three
blocks per round, four attention heads, and geometry/event-dependent bias.
Eight d3 specialists have 8,449,990 parameters each; two d5 specialists have
8,456,070 each; total 84,512,060. Exactly one specialist serves each shot.
Weights are plain FP32 NumPy arrays, with BF16 compute in the frozen configuration.

Overall P = 1 minus balanced logical failure: average durations in each
geometry/basis group, groups within distance, and give d3/d5 equal weight.
The task reward is P. A diagnostic normalized score = max((P-L)/(U-L),0) (no upper clip), with fixed L=0.7160463461538461 from
MWPM and U=0.7615159615384615 from the strongest same-test released
tensor-network predictions. TN is a rescored historical reference, not a
newly reimplemented decoder. The reference AlphaQubit gives P=0.7550607692307693
and normalized score=0.8580328368056459. Raw/unclipped scores and scientific LER,
error suppression and calibration metrics remain available.

The original accepted neural training/replay cost was about 200.3 allocated
GPU-hours on RTX PRO 6000 Blackwell Max-Q 96 GB hardware; this is not an H100
throughput claim. Agents train from scratch. Task v1.2 allows 8 H100 GPUs ×
48 hours; the unchanged ten-specialist recipe is estimated at 36–40 hours in
two waves, pending completion of the independent H100 training run. The pinned
v1.1 data payload is unchanged; task resource policy is maintained in Git.

## Reproduce

See the [GitHub task](https://github.com/T0-RSI/ai4sci-tasks/tree/main/surface-code-decoding)
for `environment/data/materialize.py`, the immutable HF lock, Harbor environments,
strict submission schema and trusted verifier. Normal deployment downloads
these pre-materialized archives; it does not regenerate experimental data.
Use the full commit revision and archive hashes pinned in GitHub. Agent data,
inference inputs and trusted labels must be mounted separately. The verifier
runs without network, replays a relocated trained model, seals outputs, and
scores them in a separate label-holding service.

Do not compare the normalized scalar as a universal measure of scientific
difficulty across tasks. It measures progress between declared task-specific
references and is not clipped above the upper reference (a score above 1 means a
decoder better than the released TN predictions). The scientific evaluation is limited to this historical
hard-readout d3/d5 distribution.

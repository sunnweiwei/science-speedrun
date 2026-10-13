# Fourier-DeepSets retrieval starter

The native modern research path is in [`glacier/README.md`](glacier/README.md).
Use this small model to establish a complete package early and as the initial
all-condition fallback; then improve either part of the native pipeline.

This is a compact, package-oriented adaptation of MassSpecGym's official
DeepSets + Fourier features baseline. It trains from scratch and has no external
weight dependency. The Fourier feature construction and baseline design are
adapted from the MIT-licensed MassSpecGym implementation; see the task's
`docs/UPSTREAM_LICENSES.md` (task repository).

The model's 10–1,000 Da preprocessing window is narrower than the input
contract. A spectrum with no retained positive peaks uses its precursor token
and zero padding. RDKit versions also disagree on some legacy candidate
valences in the pinned data: an unparseable candidate receives a zero
fingerprint and remains in the complete ranking, with ties resolved by candidate
ID. Training structures remain strictly validated. These rules do not remove
queries or candidates or change the task's chemical identity convention.

This package is the fast contract and plumbing starter, not the scientific
reference target. The reference to beat is the GLACIER-no-CF/strict-clean-MIST
hybrid documented in `/workspace/references/README.md`; it reaches 49.391%
Hit@1 over all 17,556 test queries. The former standalone
clean MIST target reaches 39.872% Hit@1 and remains the hybrid's fallback on
conditions outside GLACIER's released domain.

## Full starter run

Inside the development environment:

```bash
cp -a /workspace/baseline/starter/. /workspace/submission/
cd /workspace/submission
python train.py --input-root /workspace/data --epochs 50 --batch-size 256 --workers 8
```

The upstream v1.5 configuration uses one GPU, seed 0, learning rate 3e-4,
hidden width 128, four MLP layers, dropout 0.1, and 50 epochs. This adaptation
selects the checkpoint by validation fingerprint cosine loss so model selection
does not repeatedly rank every validation candidate pool.

Every validation improvement atomically publishes an immutable model generation
under `.versions` and switches the root `model_manifest.json`. Copy the entire
submission including dotfiles. Optimizer, scaler, RNG, data identity and elapsed
time are saved after every epoch in `/workspace/research/submission-training/last.pt`
(or the explicit `--work-dir`). Resume with the original arguments plus
`--resume /workspace/research/submission-training/last.pt`; `--epochs` and `--max-seconds`
are cumulative limits. The wall-time stop is checked after each full epoch.
Keep training state outside the submission and use a fresh work directory for
a changed model or data configuration.

## Commissioning result

The task-owned implementation was trained from scratch on the complete pinned
fold on 2026-09-02. One NVIDIA RTX A5000, 12 CPU workers, seed 0, batch size
256, and 50 epochs took 11 minutes 9 seconds. The selected epoch was epoch 2
with validation fingerprint cosine loss 0.567595. The resulting 7,500,528-byte
checkpoint has SHA-256
`176930cc9ddff4e7b1c8b226e00ac145930462e7bd9e935a398e6d0ff9975fc3`.

An operator-side replay ranked every candidate for all 19,429 validation and
17,556 test spectra without a malformed or missing output:

| Fold | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 2.800% | 10.479% | 27.176% | not run |
| Test | 5.126% | 12.406% | 27.108% | 22.185 |

Full-fold inference took 4 minutes 36 seconds for validation and 4 minutes 17
seconds for test on one A5000. The test ranking file SHA-256 is
`b2db61988686f93a8faac14f15d94b44a11cebdc6f8fb34706d010673605e213`.
MCES evaluation deduplicated 17,556 spectrum-level comparisons into 6,609
unique structure pairs and took 2 minutes 29 seconds on 32 CPU cores.
These figures commission the reproducible starter, not the strong scientific
reference. They are not substituted for an isolated verifier replay or used as
an answer-bearing artifact.

## Small plumbing run

This produces real weights without claiming a benchmark result:

```bash
python train.py \
  --input-root /workspace/data \
  --epochs 3 \
  --max-train-records 20000 \
  --max-validation-records 2000
```

Relocate `/workspace/submission`, stage validation queries using the helper
below, run `infer.sh`, and score the resulting rankings. A full contribution
smoke report should record GPU type, runtime, seed, subset sizes, validation hit
rates, and a deliberately broken-output check.

## What to improve

The starter deliberately leaves substantial room:

- richer peak-set, transformer, or graph-conditioned spectrum encoders;
- adduct, instrument, collision-energy, and neutral-loss features;
- contrastive spectrum/molecule training rather than fingerprint regression;
- learned molecular encoders and candidate-aware reranking;
- multi-spectrum aggregation without leaking across official folds;
- permitted, disjoint pretraining with fully documented provenance.

Keep the frozen `infer.sh INPUT_ROOT OUTPUT_ROOT` ranking contract even if the
architecture and training pipeline are replaced completely.

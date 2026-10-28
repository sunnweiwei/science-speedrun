# RSI experiment (author record)

The author ran one real agent research session on this task before maintainer
acceptance. Its result and recipe are kept here, outside the agent workspace, so
that later agent runs start from the same baselines. Scores are full private-test
primary metrics in percent; the exact values are in [rsi-results.json](rsi-results.json).

RSI experiment and fixed-recipe replication:

| Model | Protocol | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Agent: shared_antidrop180 | Shared, scratch; seed 0 | 24.24 | 65.62 | 65.24 | 77.84 | 77.49 |
| Agent: fixed-recipe replication | Shared, scratch; seeds 0–4 | 23.00 ± 1.71 | 62.66 ± 2.13 | 59.24 ± 7.24 | 75.35 ± 1.65 | 77.73 ± 0.75 |

## Setup

- **Agent experiment:** one real 40-hour continuing Codex research session,
  at most eight independent one-GPU training workers. It used only training and
  validation feedback and the audited initialization allowlist. The selected
  model initializes from scratch. Its recorded final training used seed 0,
  batch 64, learning rate 3e-4, weight decay .01, 25 × 2,000 = 50,000 optimizer updates and full validation, with
  task-dependent sampling and additional gradient evaluations. The shared architecture and auxiliary training state passed source review;
  five fixed-recipe seeds were then independently retrained and tested. Search cost,
  batch size and training compute differ from the reference baselines; equal
  optimizer-update counts do not establish compute-matched comparisons.

## RSI recipe and reproduction

The selected `shared_antidrop180` package and five independent fixed-recipe
retrainings are indexed in [rsi-results.json](rsi-results.json). Seed 0
reproduced the original weights byte-for-byte and all five test scores exactly.
No seed was discarded; AD65 shows substantial seed variability. These are five
retrainings of one selected recipe, not five independent searches.

Source review found one shared spectral-MLP/temporal-CNN backbone: 707,936 shared
parameters and 13,622 parameters in the five shallow adapters/offsets/heads.
Training reads only train/validation data; teacher banks and auxiliary state
are training-only and absent from inference. The selected model starts from
scratch. Architecture/export checks and 14 auxiliary fixtures passed; CUDA
fixtures passed in the portable training runtime, not in the GPU-inaccessible
Docker audit runtime. Detailed audit evidence is retained in the archive below.

Download the frozen candidate source from the pinned model release. Set
`RSI_SOURCE_DIR` to that source tree and `TASK_DATA_DIR` to development data only.
The `covariance` model belongs to this frozen source, not the baseline registry.
In the pinned training environment:

```bash
cd "$RSI_SOURCE_DIR"
for seed in 0 1 2 3 4; do
  python environment/baseline/braindecode/train_shared.py \
    --model covariance --seed "$seed" --data "$TASK_DATA_DIR" \
    --output "$TASK_RESEARCH_DIR/rsi-replication/seed-$seed/submission" \
    --epochs 25 --steps-per-epoch 2000 --batch-size 64 \
    --lr 0.0003 --weight-decay 0.01 --num-workers 0
done
```

Selection uses the equal mean of five complete validation primary metrics.
Freeze every seed's package before isolated private testing; no best-seed
selection is used. Training took 1,118–1,370 seconds per seed, excluding setup
and scoring. The 50,000 optimizer updates include 77,000 backward calls
(50,000 clean plus 27,000 SAM) and 116,000 gradient-bearing encoder views;
optimizer-update counts alone do not establish equal compute.

The selected package's independently replayed validation mean is 0.6609451645;
the training-controller record is 0.6609693717. Both are selection diagnostics,
not an official scalar test reward. The recipe-specific
[audit helper](https://github.com/T0-RSI/ai4sci-tasks/blob/4f653a77e2061ef3b723ab79c5e142d51f19f8ae/brain-computer-interface-decoding/scripts/audit_rsi_package.py)
is retained at the original source commit with the historical evidence. It
requires the frozen candidate source and model package; its optional helper
fixtures also require CUDA.

## Verification and evidence

These are author-run results; independent maintainer replay remains pending.
The final pinned Dockerfile builds passed 22 protocol tests, five isolation/output
probes, full runtime integrity and pretrained-initialization checks. In that
runtime, 125 specialist packages plus the shared BIOT seed-0 package completed
130 full track evaluations, with matching scores, coverage and hashes.
Shared-reference and RSI replay records cover 15 and six packages respectively;
these records are distinct from the final-container replay subset.

[Build, integrity and frozen-package replay commands](../tests/README.md)
and [pinned data/model downloads](../environment/data/MATERIALIZE.md#pinned-hugging-face-releases)
provide the reproduction workflow. Use the shared summarizer with
`--require-complete` for full campaigns. Shared EEGNet/ContraWR runs remain
auxiliary; the primary specialist and shared-pretrained protocols differ.
Agent submissions still require one shared backbone across all five tracks.
Public pretraining overlap is disclosed in the [provenance audit](../environment/references/PRETRAINED_AUDIT.md).

Full receipts, confusion matrices, per-file hashes, source audit and historical
smoke records remain available in the immutable
[evidence archive at commit 4f653a7](https://github.com/T0-RSI/ai4sci-tasks/tree/4f653a77e2061ef3b723ab79c5e142d51f19f8ae/brain-computer-interface-decoding/docs).
The compact JSON preserves 125 specialist runs, 15 shared packages, the selected
RSI package and five replications (230 full-track scores). `subject_count` is
an evaluator grouping count; TUEV's single group is not a patient count.
Weights and frozen candidate source are in the private HF model release;
raw campaign logs remain in operator storage.


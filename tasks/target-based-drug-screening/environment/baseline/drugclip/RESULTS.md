# Recorded reproduction results

## Current primary score: dataset-balanced BEDROC 0.166385

Protocol `drugclip-dataset-bedroc/v2` averages target BEDROC(alpha=80.5)
within each dataset, then gives DUD-E and LIT-PCBA equal weight. The two
independent epoch-14 replays yield 0.166384906515 (Harbor) and 0.166385032576
(upstream). Their raw predictions and metrics are unchanged. This is a
re-aggregation, not retraining. [Versioned replay record](reproduction/independent_replay_v2.json)

The original records below retain the earlier EF-derived v1 reward where
mentioned. That reward is not comparable to the current BEDROC score.

## GPU-trained reference: final held-out EF1% 16.108

Zifeng Wang supplied this PR #94 baseline verification on 2026-10-15.
The [reproduction record](reproduction/README.md) preserves the reported
recipe and environment; [development_bedroc.csv](reproduction/development_bedroc.csv)
contains all 34 fine-tuning epochs. These results were supplied by the run
operator, not rerun during this documentation update.

Data: `zifeng-ai/target-based-drug-screening` at
`c8d2febc996e4c96b10f3aecd89fb7375150449f`. Python 3.11; PyTorch 2.6.0+cu124.

- Dual Uni-Mol, 95,305,418 parameters; Uni-Mol v0.1 molecule initialization
  (`mol_pre_no_h_220816.pt`), random pocket initialization.
- ProFSA: 500,000 pairs, 100 completed epochs on 8 A100s in 19.0 h,
  per-GPU batch 48, LR 1e-4, molecule backbone frozen.
- BioLiP: 4 A100s, per-GPU batch 48, LR 1e-3, dev BEDROC(alpha=80.5)
  selection, patience 20. Stopped at epoch 34 in 19.6 h; selected epoch 14
  with dev BEDROC 0.4288. The early-stopping criterion completed; this does
  not mean 200 fine-tuning epochs were executed.

| Benchmark | Targets | ROC-AUC | BEDROC(80.5) | EF1% |
| --- | ---: | ---: | ---: | ---: |
| DUD-E | 102 | 0.769 | 0.295 | 18.056 |
| LIT-PCBA | 15 | 0.577 | 0.038 | 2.861 |
| Combined equal-target mean | 117 | — | — | **16.108** |

This is one dev-selected epoch-14 checkpoint across both benchmarks. Its
initial held-out EF1% of 1.830 is superseded by the rescore after correcting
pocket element tokens, the 511-atom test crop, DUD-E median/MAD normalization
then max pooling, and LIT-PCBA max pooling. It is not a second training run.
Random ranking has expected EF1%=1. The historical v1 reward computed from the rounded
16.108 is approximately 0.94155; no isolated-verifier reward file was supplied.
Other EF cutoffs and per-target metrics were not included in the report.

## Independent replay of the selected checkpoint

On 2026-11-04, the controller reproduced the epoch-14 scores on all 117
targets and 4,241,148 ligand records. The independently measured EF1% was
18.055988235982 on DUD-E, 2.861166091213 on LIT-PCBA, and
**16.107934114858** overall. The verified legacy v1 reward was **0.9415475887803699**.

The controller used the released weights with the repository's inference
code and default pocket-crop seed 1. One run used the upstream verifier.
A second run used Harbor 0.23 and the pinned task image. Both ran offline
on an RTX A6000, with UID/GID 65534 inference and independently shuffled
molecules. Per-target EF results agreed to floating-point precision.

The release's SHA256, revision, dataset identity, six metrics and comparison
of the two replays are recorded alongside the original training report.
The original training seed and code revision remain unavailable.
[Independent replay record](reproduction/independent_replay.json)

## Historical smoke, pilot, and author-checkpoint controls

These earlier experiments come from the maintained reference README and PR #94.
They are distinct from the GPU-trained reference above.

| Experiment | Training | Evaluation | ROC-AUC | BEDROC(80.5) | EF1% |
| --- | --- | --- | ---: | ---: | ---: |
| Wiring smoke, 2026-10-06 | 8 ProFSA + 8 BioLiP records, 2 epochs each | DUD-E aa2ar, 32,010 ligands | 0.34113 | 0.00466 | 0.20689 |
| Pilot, 2026-10-06 | 512 ProFSA × 3 epochs; 2,048 BioLiP × 6; 256 dev | DUD-E aa2ar, 32,010 ligands | 0.72387 | 0.02858 | 0.62066 |
| Author-checkpoint control, 2026-10-12 | Released benchmark-trained DUD-E weights | All 102 DUD-E targets | 0.77465 | 0.40126 | 24.8276 |
| Author-checkpoint control, 2026-10-12 | Released benchmark-trained LIT-PCBA weights | All 15 LIT-PCBA targets | 0.59540 | 0.07295 | 5.3661 |

The two trained runs used one RTX 3070 (8 GiB). The smoke took 239 seconds
and peaked at 1.95 GB allocated CUDA memory; the pilot took 1,398 seconds
and peaked at 4.48 GB. Both record formal_recipe=false. The pilot selected
fine-tuning epoch 2 by dev BEDROC. Its weak one-target EF1% remains below 1.

The author control's combined equal-target EF1% is 22.3325. Its weights were
trained on benchmark-related populations and are excluded from the agent
workspace and candidate initialization. It validates the evaluator, not
training/generalization in this task. Tie handling and full-target coverage
are tightened in the agent integration; historical predictions are not
available here for re-scoring.

The historical smoke/pilot recipes and checkpoint hashes remain in
[TRAINING.md](TRAINING.md). Their tiny-data scores
are not the current full-data reference score. The released-checkpoint
control is an evaluator check, not candidate training or initialization.

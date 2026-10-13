# PR summary

This track now has one human 1-Mb protocol with eleven equally weighted output
families, 3,200/200/1,000 examples, all 5,930 real tracks, a 100-GiB dataset
cap, and a 50M learned-parameter model cap.

## Leakage-free splicing evaluation

- Inference receives only DNA, window coordinates, padding, and the public
  junction strand mapping. Real splice positions, target references, labels,
  and scoring queries remain on the trusted side.
- Junction rows follow donor-positive, acceptor-positive, donor-negative,
  acceptor-negative order. Track strand comes from `junction_channel_half`.
- Models emit complete usage predictions, four model-selected site lists, and
  the corresponding junction cube. The grader matches trusted coordinates to
  predicted endpoints; a missing endpoint is a zero prediction and never
  removes a scoring sample.
- The frozen scoring set contains the 8,192 deterministic compact queries plus
  at most one observed nonzero event per available track/example. Training,
  validation checkpoint selection, and formal grading all consume the same
  generated scoring parts. The audited supervision ceiling is 626/734 junction
  tracks; the other 108 are not manufactured as negatives.
- A varying target with constant predictions receives Pearson 0 rather than
  being dropped from the track or family mean.

## Isolation and sealing

- Development mounts only train/validation labels, scoring parts, targets,
  public inputs, and reference data. Formal inference mounts only public test
  inputs and the reference. The trusted grader alone mounts test labels,
  scoring parts, and targets.
- Development, baseline training/evaluation, formal inference, and grading are
  explicitly network-disabled. Formal inference reserves one NVIDIA GPU and
  verifies CUDA after demotion to uid 10001.
- After inference exits, the verifier kills and confirms termination of every
  uid-10001 process, including detached descendants. It then copies a
  root-owned, read-only snapshot, validates that final snapshot, hashes it, and
  atomically publishes `READY`.
- The baseline cosine schedule always uses the complete requested epoch count;
  per-epoch checkpoint selection uses an independent `stop_after_epoch` and
  never restarts the schedule. Only complete-epoch `last.pt` checkpoints may
  resume training; `progress.pt` records intermediate diagnostics.

## Current release layout

The installed release uses one layout: `targets`, `metadata`, `splits`,
`public`, `labels`, `scoring`, root `release.json`, and `reference`. The active
materializer writes this layout directly and the shared `export_release()`
builds public, label, and scoring manifests.

The legacy-layout migration performs no download, TFRecord read, target decode,
or target recomputation. It moves existing stores, rewrites manifest and
receipt paths, preserves scoring parts, and compares SHA-256 for every
manifest-referenced target store before and after the move. Serialized schema
versions remain explicit compatibility identifiers and are rejected when
incompatible; they are not selectable task profiles.
Migration persists a target hash journal before moving files and resumes from
either legacy or current target paths after an interruption.

## Validation status

The complete materialized release contains 89.22 GiB of referenced targets and
92.49 GiB installed data. The current Hub release is
`release-44d06e9702eba731`, pinned at commit
`1978bcafbea5d8b09ba9f1821ee4a71a4331e05f`. The public-only GPU smoke and sealed eleven-family
grader replay validate the interface, but their random tiny checkpoint is not a
trained baseline. Only a complete trained-model run from the independent
trusted grader should be recorded as a benchmark score.

## Tiny overfit on the materialized release

A 226,248-parameter version of the reference model completed 120 GPU steps on
the first real materialized training example while retaining the 1-Mb input and
all eleven output families. The fixed-example supervised loss decreased from
5.3842 to 0.8150 (84.86%). Reproduction instructions and the local checkpoint
path are in `../TRAINING.md`; the committed per-family result is in
`../results/tiny_overfit_2026-09-30.json`. This is a training-path diagnostic,
not a held-out evaluation result.

## Full-data tiny model: three epochs and held-out test

The 226,248-parameter configuration trained for three complete epochs over
all 3,200 materialized train examples (9,600 optimization steps). Validation
used all 200 examples; the best checkpoint was epoch 3, selected before the
1,000-example held-out test evaluation. The comparable validation mean-family
Pearson scores were **0.09462** at epoch 2 and **0.12863** at epoch 3. The
local held-out test mean-family Pearson was **0.14236**.

| Family | Best validation | Held-out test |
| --- | ---: | ---: |
| ATAC | 0.20389 | 0.25845 |
| DNase | 0.37744 | 0.41479 |
| PRO-cap | 0.18590 | 0.17252 |
| CAGE | 0.06596 | 0.06049 |
| RNA-seq | 0.12398 | 0.11174 |
| TF ChIP-seq | 0.17216 | 0.20312 |
| Histone ChIP-seq | 0.15860 | 0.17683 |
| Splice sites | -0.00559 | -0.00476 |
| Splice-site usage | 0.02809 | 0.02400 |
| Splice junctions | 0.00000 | 0.00000 |
| Contact maps | 0.10455 | 0.14878 |

The epoch-1 validation score (0.00027) used an earlier mixed-precision metric
path; epoch 2 was recomputed with the corrected evaluator before final
checkpoint selection. Local evaluation uses FP32 inference, FP16 output
rounding, predicted splice positions, and trusted hidden queries. It is a
held-out diagnostic, not the sealed inference and grader replay. Full per-track
reports, dataset-release hash, and checkpoint hash are recorded in
`../results/tiny_full_3epoch_summary_2026-09-30.json` and its linked JSON
reports. The reproducible command and local checkpoint path are in
`../TRAINING.md`.

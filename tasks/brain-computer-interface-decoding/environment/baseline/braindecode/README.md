# Braindecode 1.8.1 baselines

[RESULTS.md](RESULTS.md) has the five-seed test scores and training settings of every model
below; [RESULTS.json](RESULTS.json) has each seed's five scores and package hashes. Two kinds of
reference were trained:

- **Shared references** (CBraMod, BIOT and CodeBrain, pretrained): one backbone per family and
  seed, trained jointly on all five tracks. This is the form a submission must take.
- **Single-task specialists** (EEGNet, ContraWR, CBraMod, BIOT and CodeBrain, from scratch): one
  model per track, so they are not valid submissions. They show what each track's data supports
  on its own.

| file | role |
| --- | --- |
| `shared.py`, `train_shared.py` | shared model (backbone, per-track channel adapters and heads) and its joint training |
| `shared_infer.py`, `shared_infer.sh` | shared-package inference; `train_shared.py` copies them into the package as `infer.sh` |
| `model.py`, `train.py`, `train_full.py`, `infer.py`, `infer.sh` | single-task specialists |
| `data.py`, `sampling.py` | data loading (divides microvolts by 100) and class-balanced sampling |
| `pretrained.py`, `pretrained.lock.json`, `validate_pretrained.py` | audited loading of the three checkpoints in `/workspace/data/pretrained/` |
| `publish.py` | atomic copy of a package to `/workspace/submission` |

## Shared model

`train_shared.py` builds one backbone per model family. Each track has a 1×1 channel adapter
into the backbone's channels and its own classification head. The backbone sees 2-second chunks;
a final partial chunk is zero-padded and weighted by its observed duration when chunk features
are pooled. Training cycles through the five tracks evenly, with each track's iterator carried
across reporting epochs. After every reporting epoch the model is scored on all five validation
splits, and the checkpoint with the best equal-weight mean of the five validation scores is
exported as a package (`infer.sh`, code, `config.json`, `weights.pt`, `training.json` with the
validation history).

A pilot run checks the whole pipeline in a few minutes:

```bash
python /workspace/baseline/braindecode/train_shared.py --model eegnet --seed 0 \
  --data /workspace/data --output /workspace/research/shared/eegnet-pilot/submission \
  --epochs 1 --steps-per-epoch 100 --batch-size 8 --train-limit 1024 --val-limit 512
python /workspace/baseline/braindecode/publish.py \
  /workspace/research/shared/eegnet-pilot/submission --check
```

The shared references used the full recipe (50 reporting epochs × 1,000 updates, so each track
receives 10,000 batches):

```bash
python /workspace/baseline/braindecode/train_shared.py --model codebrain --seed 0 \
  --checkpoint-dir /workspace/data/pretrained \
  --data /workspace/data --output /workspace/research/shared/codebrain-s0/submission \
  --epochs 50 --steps-per-epoch 1000 --batch-size 8 --lr 0.0001 --weight-decay 0.05 \
  --num-workers 2
```

`--output` becomes a symlink to a versioned package directory; `publish.py` resolves it and copies
a plain directory into `/workspace/submission`, because a package must not contain symlinks.
Training resumes from `<output>-resume.pt` at completed reporting epochs with optimizer, scheduler,
RNG and sample order.

## Pretrained initialization

CBraMod, BIOT and CodeBrain load only the audited public backbone tensors pinned in
`pretrained.lock.json` (`braindecode/*-pretrained` on Hugging Face, BSD-3-Clause). The source
classification heads are discarded, and `pretrained.py` rejects a checksum mismatch or any
missing or shape-mismatched core tensor. Public pretraining corpora can overlap downstream
sources, including TUEV/TUEG; [the provenance audit](../../references/PRETRAINED_AUDIT.md) lists
the source corpora and the unresolved overlap. EEGNet and ContraWR start from scratch.

Parameter counts of the shared architectures, including five adapters and five heads: EEGNet
9,094; ContraWR 1,581,302; CBraMod 4,931,878; BIOT 3,195,798; CodeBrain 15,175,670.

## Single-task specialists

All specialist runs use seeds 0–4, random initialization, 50 complete epochs, batch 32, AdamW
(learning rate 1e-4, weight decay 0.05), a cosine schedule, label smoothing 0.1 and gradient
clipping 1. FACED and TUEV checkpoints are selected by validation Cohen κ, the other three by
their primary metric. For example:

```bash
python /workspace/baseline/braindecode/train_full.py \
  --model cbramod --track faced --seed 0 --epochs 50 \
  --batch-size 32 --lr 0.0001 --weight-decay 0.05 --num-workers 2 \
  --data /workspace/data --output /workspace/research/specialists/cbramod-faced-s0/submission/faced
```

Model-specific input settings:

- EEGNet on FACED and TUEV adds `--input-normalization per-channel-zscore` (per-sample,
  per-channel z-score over the window, in training and inference); TUEV also adds
  `--class-balanced` (inverse-frequency sampling with replacement). Other EEGNet tracks use the
  /100 scaling only.
- ContraWR on FACED adds `--normalization-layer group` (one-group GroupNorm instead of
  BatchNorm).
- `--cache-data` keeps a split in memory without changing values or order.

Selecting by κ can pick a TUEV model with lower balanced accuracy; compare families by the
primary validation metric at each saved checkpoint. Measured parameter counts of the
specialists:

| Model | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 10,553 | 4,342 | 4,387 | 2,898 | 2,418 |
| ContraWR | 1,582,569 | 1,572,582 | 1,573,539 | 1,598,050 | 1,595,746 |
| CBraMod (package) | 5,500,009 | 5,020,006 | 5,038,003 | 5,023,202 | 4,970,402 |
| BIOT | 3,192,585 | 3,187,718 | 3,187,715 | 3,198,466 | 3,197,442 |
| CodeBrain | 66,530,409 | 28,129,806 | 45,729,203 | 55,009,002 | 33,889,002 |

The counts vary with the task shape; this CodeBrain configuration is much larger than the
single figure in the literature.

## Scoring on validation

`/workspace/evaluation/bci_score.py --track TRACK --truth TRUTH --predictions PREDICTIONS
--output OUT` applies the verifier's metric to a predictions file. The truth file is JSON lines
with `id`, `label`, `subject` and `session`, taken from a validation manifest. `train_shared.py`
already reports all five validation scores after every reporting epoch.

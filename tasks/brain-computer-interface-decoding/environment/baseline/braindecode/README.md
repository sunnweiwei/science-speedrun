# Braindecode 1.8.1 baselines (V4)

[RESULTS.md](RESULTS.md) records reference test scores and training settings; [RESULTS.json](RESULTS.json)
retains per-seed historical records and package hashes. Two kinds of authorized reference are supported
under the active 4-track V4 protocol (`tuev`, `ad65`, `openbmi`, `longitudinal`):

- **Shared references** (`cbramod`, `codebrain`, pretrained; or `eegnet`, `contrawr` from scratch):
  one backbone per family and seed, trained jointly on all four active EEG tracks (`tuev`, `ad65`,
  `openbmi`, `longitudinal`). This is the form a submission must take.
- **Single-task specialists** (`eegnet`, `contrawr`, `cbramod`, `codebrain`, from scratch): one
  model per track, so they are not valid submissions. They indicate what each track's training data
  supports on its own.

> **V4 De-Contamination Note:** `FACED` is removed from active tracks due to recurring video clip
> stimulus/EOG confounds across subjects, and `BIOT` (`braindecode/biot-pretrained-six-datasets-18chs`)
> is removed from authorized models/checkpoints (`pretrained.lock.json` `disallowed_models`) due to
> direct `TUEV` pretraining corpus overlap.

| file | role |
| --- | --- |
| `shared.py`, `train_shared.py` | shared model (backbone, 4 per-track channel adapters and heads) and its joint training |
| `shared_infer.py`, `shared_infer.sh` | sample-local shared-package inference compatible with per-subject/session isolated verification |
| `model.py`, `train.py`, `train_full.py`, `infer.py`, `infer.sh` | single-task specialists |
| `data.py`, `sampling.py` | data loading (divides microvolts by 100) and class-balanced sampling |
| `pretrained.py`, `pretrained.lock.json`, `validate_pretrained.py` | audited loading of the two allowlisted checkpoints in `/workspace/data/pretrained/` (`cbramod`, `codebrain`) and active blocking of `biot` |
| `publish.py` | atomic copy of a package to `/workspace/submission` |

## Shared model

`train_shared.py` builds one backbone per model family (`eegnet`, `contrawr`, `cbramod`, `codebrain`).
Each of the four active tracks (`tuev`, `ad65`, `openbmi`, `longitudinal`) has a 1×1 channel adapter
into the backbone's channels and its own classification head. The backbone sees 2-second chunks;
a final partial chunk is zero-padded and weighted by its observed duration when chunk features
are pooled. Training cycles through the four tracks evenly, with each track's iterator carried
across reporting epochs. After every reporting epoch the model is scored on all four validation
splits, and the checkpoint with the best V3/V4 chance-adjusted validation composite across the four
tracks is exported as a package (`infer.sh`, code, `config.json`, `weights.pt`, `training.json`).

A pilot run checks the whole pipeline in a few minutes:

```bash
python /workspace/baseline/braindecode/train_shared.py --model eegnet --seed 0 \
  --data /workspace/data --output /workspace/research/shared/eegnet-pilot/submission \
  --epochs 1 --steps-per-epoch 100 --batch-size 8 --train-limit 1024 --val-limit 512
python /workspace/baseline/braindecode/publish.py \
  /workspace/research/shared/eegnet-pilot/submission --check
```

Full shared CodeBrain training recipe across the four V4 tracks:

```bash
python /workspace/baseline/braindecode/train_shared.py --model codebrain --seed 0 \
  --checkpoint-dir /workspace/data/pretrained \
  --data /workspace/data --output /workspace/research/shared/codebrain-s0/submission \
  --epochs 50 --steps-per-epoch 1000 --batch-size 8 --lr 0.0001 --weight-decay 0.05 \
  --num-workers 2
```

`--output` becomes a symlink to a versioned package directory; `publish.py` resolves it and copies
a plain directory into `/workspace/submission`, because a package must not contain symlinks.

## Pretrained initialization

`cbramod` and `codebrain` load only the audited public backbone tensors pinned in
`pretrained.lock.json` (`braindecode/*-pretrained` on Hugging Face, BSD-3-Clause). The source
classification heads are discarded, and `pretrained.py` rejects `biot`, any checksum mismatch, or
any missing or shape-mismatched core tensor. See [PRETRAINED_AUDIT.md](../../references/PRETRAINED_AUDIT.md)
for the full provenance audit. `eegnet` and `contrawr` initialize from scratch.

## Single-task specialists

Single-task specialists train from random initialization on one active track (`tuev`, `ad65`,
`openbmi`, or `longitudinal`), e.g.:

```bash
python /workspace/baseline/braindecode/train_full.py \
  --model cbramod --track tuev --seed 0 --epochs 50 \
  --batch-size 32 --lr 0.0001 --weight-decay 0.05 --num-workers 2 \
  --data /workspace/data --output /workspace/research/specialists/cbramod-tuev-s0/submission/tuev
```

## Scoring on validation

`/workspace/evaluation/bci_score.py --track TRACK --truth TRUTH --predictions PREDICTIONS
--output OUT` applies the verifier's subject/session-stratified metric to a predictions file for
any of the four active tracks (`tuev`, `ad65`, `openbmi`, `longitudinal`).

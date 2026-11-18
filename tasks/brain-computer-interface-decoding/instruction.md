# Decode four EEG tasks with one shared model

## Task and goal

Develop one EEG model that decodes four heterogeneous clinical and brain–computer interface (BCI)
recording cohorts:

- clinical EEG events (`tuev`, 6 classes, 16 bipolar channels, 5 s windows);
- Alzheimer's disease, frontotemporal dementia or healthy control (`ad65`, 3 classes, 19 channels, 10 s);
- left or right motor imagery (`openbmi`, 2 classes, 62 channels, 4 s trials);
- eyes closed or open in a longitudinal resting-state study (`longitudinal`, 2 classes, 58 channels, 2 s).

One trainable backbone serves all four tracks and is trained jointly on all four training splits.
Each track may have its own input adapter and classification head. A decoder that generalizes across
devices, montages, paradigms and unseen subjects/patients is the core objective of EEG foundation
models for clinical neurophysiology and brain–computer interfaces.

Every track is evaluated on held-out subjects/patients (`tuev`: the official patient-disjoint
evaluation set) using **subject/session-stratified balanced accuracy** `BA_k` in `[0, 1]`:

- `tuev` (`K = 6`, `chance = 1/6`): balanced accuracy across the official patient-disjoint evaluation set;
- `ad65` (`K = 3`, `chance = 1/3`): patient-level balanced accuracy after averaging window probabilities within each patient;
- `openbmi` (`K = 2`, `chance = 0.5`): balanced accuracy computed within each held-out subject, then averaged equally across subjects;
- `longitudinal` (`K = 2`, `chance = 0.5`): balanced accuracy computed within each recording session, averaged across sessions within each subject, then averaged equally across subjects.

The primary benchmark score (`reward.json["score"]` and `reward.json["reward"]`) is the V3/V4
equal-weight chance-adjusted skill across the four tracks, clipped to `[0, 1]`:

- `skill_k = (BA_k - chance_k) / (1 - chance_k)` for `k` in `("tuev", "ad65", "openbmi", "longitudinal")`
  with `chance = {"tuev": 1/6, "ad65": 1/3, "openbmi": 0.5, "longitudinal": 0.5}`;
- `signed_score = 0.25 * (skill_tuev + skill_ad65 + skill_openbmi + skill_longitudinal)`;
- `score = reward = clip(signed_score, 0.0, 1.0)`.

| test score, five-seed mean (%) | TUEV | AD65 | OpenBMI | Longitudinal | V3/V4 Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant class (chance) | 16.67 | 33.33 | 50.00 | 50.00 | 0.0000 |
| shared CBraMod, pretrained | 40.71 | 51.65 | 57.41 | 74.54 | 0.3006 |
| shared CodeBrain, pretrained (baseline) | 55.77 | 59.75 | 63.31 | 76.89 | 0.4174 |
| best single-task specialist per track | 61.36 | 49.02 | 77.30 | 76.03 | 0.4596 |

The shared models follow the submission rules. Single-task specialists are trained on one track
only and are not valid submissions; they indicate per-track reference capacity.

## Workspace

- `/workspace/contract.md`: the package schema, `infer.sh` entry point, input and output formats,
  per-subject/session isolated inference rules, and resource limits.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `tuev/`, `ad65/`, `openbmi/` and `longitudinal/` each hold `train/` and `val/`:
    NumPy shards in microvolts and a manifest with labels, subjects and sessions;
  - `pretrained/` holds the two audited initialization checkpoints (`bci-cbramod.safetensors`,
    `bci-codebrain.safetensors`).

  All of these are read-only.
- `/workspace/baseline/braindecode/`: the Braindecode 1.8.1 baselines: shared training
  (`train_shared.py`), single-task trainers, inference code and `publish.py`; `RESULTS.md`
  and `RESULTS.json` with reference scores and training settings; `reproduction/`
  with validation histories.
- `/workspace/references/`: the provenance audit of pretrained checkpoints (`PRETRAINED_AUDIT.md`).
- `/workspace/evaluation/`: the verifier's scoring code (`bci_score.py`) for scoring validation
  predictions.
- `/workspace/submission/`: your package (empty at start).
- `/workspace/research/`: writable space for derived data, checkpoints and notes.

## Rules

- Initialize from scratch or ONLY from the audited checkpoints in `/workspace/data/pretrained/`
  pinned in `/workspace/baseline/braindecode/pretrained.lock.json` (`cbramod`, `codebrain`).
  Checkpoints with direct downstream `TUEV` pretraining exposure (such as `BIOT`) or any other
  external weights/datasets are strictly prohibited and rejected by the verifier.
- All four training splits may update the shared backbone. Use the validation splits only for
  model selection.
- Test data are off limits; they are not in this container.
- **Per-subject / per-recording-session isolated inference**: during blind verification, `infer.sh`
  is invoked independently on each held-out `(subject, recording_session)` unit with a freshly
  staged shard containing only that unit's windows. Do not train, adapt, or rely on cohort-wide
  transductive feature whitening or cross-subject batch normalization statistics at inference time.
- You may change the architecture, preprocessing, sampling, losses and optimization.
- There is no network.

## Submission

Keep one complete package at `/workspace/submission` that follows `contract.md` from early on:
`infer.sh`, code, configuration and weights for the shared backbone with its four adapters and
heads. The package must run on unseen inputs after relocation; a score or a predictions file is
not a submission. Replace it atomically whenever you have a better model: stage it elsewhere,
then swap it in, as `python /workspace/baseline/braindecode/publish.py PACKAGE --check` does.

## Continuous research

The baselines have already been trained and scored: `/workspace/baseline/braindecode/RESULTS.md`
records their test scores and settings. Do not retrain a baseline merely to confirm its numbers;
check your own pipeline with a short pilot run instead.

There is no end point. Identify the tracks where the shared model lags, form a hypothesis, train
and validate a candidate across all four validation splits, and publish it when its validation
aggregate improves. Then start the next experiment.

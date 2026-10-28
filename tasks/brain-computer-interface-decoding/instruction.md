# Decode five EEG tasks with one shared model

## Task and goal

Develop one EEG model that decodes five kinds of recordings:

- emotion while watching video clips (FACED, 9 classes, 32 channels, 10 s windows);
- clinical EEG events (TUEV, 6 classes, 16 bipolar channels, 5 s);
- Alzheimer's disease, frontotemporal dementia or healthy control (AD65, 3 classes, 19 channels, 10 s);
- left or right motor imagery (OpenBMI, 2 classes, 62 channels, 4 s trials);
- eyes closed or open in a longitudinal resting-state study (2 classes, 58 channels, 2 s).

One trainable backbone serves all five tracks and is trained jointly on all five training splits.
Each track may have its own input adapter and classification head. A decoder that holds across
devices, montages, tasks and new people is what EEG foundation models aim for, and brain-computer
interfaces and clinical EEG reading depend on it.

Every track is tested on held-out subjects (TUEV: the official evaluation set) and scored in
[0, 1] by its own metric:

- FACED and TUEV: balanced accuracy over windows;
- AD65: balanced accuracy over patients, after averaging each patient's window probabilities;
- OpenBMI: balanced accuracy within each subject, averaged over subjects;
- Longitudinal: balanced accuracy within each session, averaged within each subject, then over
  subjects.

The five scores have equal standing and are reported separately; there is no aggregate. The goal
is to raise all five.

| test score, five-seed mean (%) | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant class (chance) | 11.11 | 16.67 | 33.33 | 50.00 | 50.00 |
| shared CBraMod, pretrained | 14.91 | 40.71 | 51.65 | 57.41 | 74.54 |
| shared BIOT, pretrained | 16.75 | 50.69 | 59.71 | 59.12 | 74.22 |
| shared CodeBrain, pretrained (baseline) | 19.58 | 55.77 | 59.75 | 63.31 | 76.89 |
| best single-task specialist per track | 42.22 | 61.36 | 49.02 | 77.30 | 76.03 |

The shared models follow the submission rules. The specialists (CodeBrain, CodeBrain, EEGNet,
EEGNet, BIOT) are each trained on one track only, so they are not valid submissions; they show
what each track's data supports on its own.

## Workspace

- `/workspace/contract.md`: the package, the `infer.sh` entry point, input and output formats and
  the inference limits.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `faced/`, `tuev/`, `ad65/`, `openbmi/` and `longitudinal/` each hold `train/` and `val/`:
    NumPy shards in microvolts and a manifest with labels, subjects and sessions;
  - `pretrained/` holds the three audited initialization checkpoints (CBraMod, BIOT, CodeBrain).

  All of these are read-only.
- `/workspace/baseline/braindecode/`: the Braindecode 1.8.1 baselines: shared training
  (`train_shared.py`), the single-task trainers, inference code and `publish.py`; `RESULTS.md`
  and `RESULTS.json` with every seed's test scores and the training settings; `reproduction/`
  with the shared references' validation histories.
- `/workspace/references/`: the audit of the pretrained checkpoints and source notes.
- `/workspace/evaluation/`: the verifier's scoring code (`bci_score.py`) for scoring validation
  predictions.
- `/workspace/submission/`: your package (empty at start).
- `/workspace/research/`: writable space for derived data, checkpoints and notes.

## Rules

- Initialize from scratch or from the three checkpoints in `/workspace/data/pretrained/`. Do not
  use any other weights or datasets.
- All five training splits may update the shared backbone. Use the validation splits only for
  model selection.
- Test data are off limits; they are not in this container.
- Do not train or adapt on evaluation inputs at inference time.
- You may change the architecture, preprocessing, sampling, losses and optimization.
- There is no network.

## Submission

Keep one complete package at `/workspace/submission` that follows `contract.md` from early on:
`infer.sh`, code, configuration and weights for the shared backbone with its five adapters and
heads. The package must run on unseen inputs after relocation; a score or a predictions file is
not a submission. Replace it atomically whenever you have a better model: stage it elsewhere,
then swap it in, as `python /workspace/baseline/braindecode/publish.py PACKAGE --check` does.

## Continuous research

The baselines have already been trained and scored in this setup:
`/workspace/baseline/braindecode/RESULTS.md` has their test scores and settings, `RESULTS.json`
every seed, and `reproduction/` the validation history of every shared reference. Do not retrain
a baseline to confirm its numbers; check your own pipeline with a short run instead (the baseline
README has a 100-update pilot).

There is no end point. Find the tracks where the shared model falls furthest behind what the
data supports, form a hypothesis, train and validate a candidate on all five validation splits,
and publish it when it is better. Then start the next experiment. Do not stop to hand in a final
answer. A published model or a reached score is a milestone, not the end of the task.

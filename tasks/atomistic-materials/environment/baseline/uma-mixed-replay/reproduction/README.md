# Reproduction record: uma-mixed-replay

The training and validation runs behind `../RESULTS.md`. The model is `/workspace/data/models/mixed-replay/`, trained with `prepare_mixed.py` on eight H100s (2 h 17 min); it has already been trained and scored, so there is no need to train it again to see what it does.

| file | content |
| --- | --- |
| `training-preparation.json` | the frame counts and OMat24 subsets of the replay mixture |
| `train_progress.jsonl` | `VALIDATION_PROGRESS` after every epoch (energy, force and stress MAE on the held-out OMat24 and off-equilibrium frames) and `TIMING_PROGRESS` |
| `validation_scores/physical-score.json` | `evaluation/score.py` on the validation stability, phonon and elasticity panels: per metric, n, failures and the error statistics |
| `validation_scores/<suite>.score.json` | `evaluation/score_extra.py` for amorphous, cleavage, kappa, MOF and NEB on their validation inputs |
| `validation_predictions.tgz` | the raw predictions behind those scores (`val/physical/predictions.jsonl` and one directory per extra suite); `tar xzf` it under `/workspace/research` |

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

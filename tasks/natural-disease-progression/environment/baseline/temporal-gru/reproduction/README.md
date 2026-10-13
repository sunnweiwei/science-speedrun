# Reproduction record: temporal-gru

The run behind `../RESULTS.md` (seed 20260929, one H100). The baseline has already been trained and
scored, so there is no need to train it again to see what it does. Only aggregate numbers are kept
here: no patient-level predictions or outputs.

| file | content |
| --- | --- |
| `training.json` | parameters, checkpoint rule, objective, the internal split, and per epoch the training loss, internal-validation loss and macro time-averaged AUROC |
| `calibration.json` | the post-hoc calibration: strengths and Brier before and after on the reserved fold |
| `validation_panel.json` | `evaluation/score.py` on the 3,915 validation patients: the panel metrics (the reward is `panel_recall_at_5pct`) and, per endpoint, recall and PPV at 1 % and 5 %, AUROC, AUPRC, prevalence and calibration. Endpoints with fewer than 10 validation positives are left out. |

The test numbers in `../RESULTS.md` come from the verifier.

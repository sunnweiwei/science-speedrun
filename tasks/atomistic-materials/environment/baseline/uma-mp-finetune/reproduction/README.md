# Reproduction record: uma-mp-finetune

The validation run behind `../RESULTS.md`, made with `run_local.py`, `extra.py` and the task's scorers on eight H100s. The model is the 3-epoch checkpoint at `/workspace/data/models/mp-finetune-3ep/`; it has already been scored, so there is no need to run these suites again to see what it does.

| file | content |
| --- | --- |
| `validation_scores/physical-score.json` | `evaluation/score.py` on the validation stability, phonon and elasticity panels: per metric, n, failures and the error statistics |
| `validation_scores/<suite>.score.json` | `evaluation/score_extra.py` for amorphous, cleavage, kappa, MOF and NEB on their validation inputs |
| `validation_predictions.tgz` | the raw predictions behind those scores (`val/physical/predictions.jsonl` and one directory per extra suite); `tar xzf` it under `/workspace/research` |

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

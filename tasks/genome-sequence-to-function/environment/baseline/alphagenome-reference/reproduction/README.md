# Reproduction record: alphagenome-reference

The run behind `../RESULTS.md`. The baseline has already been trained and scored; these files are
its record, so there is no need to train it again to see what it does.

Run:
- the commands in `../README.md`, `reference_large.yaml`, seed 17;
- 8× H100 80 GB, 2026-10-01;
- 71,792 steps (14 epochs) in 4 h 9 min.

The checkpoint is `last.pt`, sha256 `5388434c…77ee`, 215,286,020 parameters.

| file | content |
| --- | --- |
| `train_log.jsonl` | `train_start`, one `train_step` every 200 steps (loss, gradient norm, learning rate, elapsed seconds), `train_complete` |
| `model_manifest.json` | the checkpoint's configuration and digest |
| `validation_tracks.json` | `evaluation/score.py` on all 200 validation intervals: per family, the Pearson of every track, the values per track and the missing-prediction ratio |
| `validation_variants.json` | `score_variants.py` on the development variant split: per evaluation, the rows, the metric, the model's score and the anchors |

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

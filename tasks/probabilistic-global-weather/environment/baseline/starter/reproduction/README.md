# Reproduction record: starter (native WN2, 580-update recipe)

The reviewers' run behind `../RESULTS.md`. The baseline has already been trained and scored; these
files are its record, so there is no need to train it again to see what it does.

Run:
- `train.py` with `schedule.json`, seed 0, on four H100 80 GB in Docker, 2026-09-25;
- all eleven stages passed in 13.1 h;
- 183,785,575 parameters, no pretrained parameters loaded;
- the published package scored reward 0.674377 (normalized CRPS 0.482850) in the full verifier.

| file | content |
| --- | --- |
| `run.json` | seed, schedule digest, final status and total seconds |
| `stages/stage-NN.json` | one report per stage: its schedule entry (resolution, cadence, autoregressive steps, learning rate), devices and batch, setup seconds, every logged update (global update, seconds, loss, gradient norm, learning rate, the weather windows used), wall seconds and the weights digest |

The losses are not comparable across stages: resolution, cadence and rollout length change between them.

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

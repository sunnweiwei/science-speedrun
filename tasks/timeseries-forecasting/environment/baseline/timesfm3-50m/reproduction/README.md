# Reproduction record: timesfm3-50m

The run behind `../RESULTS.md`. The baseline has already been trained and scored; these files are
its record, so there is no need to train it again to see what it does.

Run:
- the commands in `../README.md` on eight H100s, 2026-10-03;
- 150,000 steps in 3 h 10 min;
- the best validation checkpoint (step 100,000) was published and scored by the verifier (reward 0.275).

| file | content |
| --- | --- |
| `config.json` | the training arguments |
| `train.log` | loss and learning rate every 100 steps, and each validation pass |
| `validation_curve.jsonl` | every 5,000 steps: validation skill, relative MASE and WQL against seasonal naive on the 15 stratified validation configs, loss and seconds |
| `best_checkpoint.json` | the selected step and its validation skill |

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

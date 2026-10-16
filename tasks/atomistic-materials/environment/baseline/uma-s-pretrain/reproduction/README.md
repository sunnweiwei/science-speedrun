# Reproduction record: uma-s-pretrain (mixed-source scratch trial)

This is the record of the run behind `../RESULTS.md`: run `scratch-10ep-mixed1m-20261005-01`, random UMA-S trained for 10 epochs on 1,000,000 mixed-source frames with eight H100s. It has already been scored on the full validation, so there is no need to train or evaluate it again.

| file | content |
| --- | --- |
| `validation_metrics.json` | Score v2 and the 15 raw metrics under the verifier's reward-key names; case counts per suite; failure counts; start and end dev loss |
| `training-loss.png` | training and labeled-dev loss over the 312,500 updates (320 dev checks), plus per-source dev errors |

Raw validation predictions, training logs and the mixed-sample preparation script are not included. No test predictions exist for this run: its test split was not used.

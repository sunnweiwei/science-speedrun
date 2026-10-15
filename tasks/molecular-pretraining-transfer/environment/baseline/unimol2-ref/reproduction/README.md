# Reproduction record: unimol2-ref

The runs behind `../RESULTS.md`. The baseline has already been pretrained and scored; these files
are its record, so there is no need to pretrain it again to see what it does.

Runs:
- **pretraining**: `pretrain.sh` with the defaults on 8 H100, 2026-09-28: 30,000 updates in 5.1 h;
- **transfer**: the package from `checkpoint_last.pt` replayed through the verifier, 2026-09-29.

| file | content |
| --- | --- |
| `pretrain_attempts_1-5.log` | the first attempts; they hung at update 10,700 (the unequal-shard `IndexError` in `../README.md`) |
| `pretrain.log` | the run resumed from update 10,000 after the fix, to 30,000 |
| `pretrain_validation.log` | the validation passes on the 100k-molecule set: total, masked-token, coordinate and distance losses |
| `transfer_logs/<task>.log` | per downstream task, the verifier's fine-tune (`evaluation/protocol/finetune.sh`): its arguments and the validation loss and metrics after every epoch. The best validation epoch predicts the test split. |

The test numbers in `../RESULTS.md` come from the verifier; no test predictions are included.

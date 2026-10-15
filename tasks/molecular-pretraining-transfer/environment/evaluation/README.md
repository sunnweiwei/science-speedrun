# Development evaluation

The same fine-tune code as the verifier: `protocol/` is byte-identical to the verifier's copy.

| script | use |
|---|---|
| `check_submission.py [PKG]` | the verifier's package check: files, `encoder.json`, the 90 M parameter budget, and that every encoder weight is in `checkpoint.pt` |
| `dev_eval.py --encoder PKG\|none --out DIR [--tasks ...] [--gpus ...]` | runs the fixed fine-tune protocol on every task (train split, 15 epochs, best-validation epoch). It predicts the **validation** split, the development stand-in for the held-out test split, and reports the verifier's metrics. |
| `protocol/finetune.sh`, `predict.py`, `run_suite.py`, `check_encoder.py`, `tasks.py`, `unimol2x/` | the verifier's protocol: one fine-tune, test prediction, GPU scheduling, the package check, metrics, and the Uni-Core user-dir |

A full `dev_eval.py` run is 24 fine-tunes. It takes about 30 minutes on 8 H100 with three fine-tunes per GPU,
and several hours on one GPU. `--tasks` limits it to a subset for quick checks, for example
`--tasks qm9dft_v2 exrx_logd asap_logd`. Use `--encoder none` for the no-pretraining reference.

The validation split is also used to pick the epoch, so validation numbers are optimistic compared with test.
They are for comparing candidates. Write every output under `/workspace/research`.

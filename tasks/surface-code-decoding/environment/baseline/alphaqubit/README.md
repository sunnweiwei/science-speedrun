# Baseline: independent AlphaQubit reproduction, trained from scratch

This directory is a Python project, and the image sets `PYTHONPATH=/workspace/baseline/alphaqubit`:

| path | contents |
| --- | --- |
| `baselines/` | the AlphaQubit model and its training code: `alphaqubit_train` (pretraining on streaming Stim simulation from the even-fitted DEMs, then fine-tuning on the training rows) and `alphaqubit_artifact` (freezing) |
| `qecbench/` | task library: packaging (`qecbench.package`), validation staging, scoring (the verifier's code), `qecbench.train` for fine-tuning one specialist of a packaged submission |
| `reproduction/` | the ten specialist groups (`groups.txt`), the reproduction's training logs (`run/`), result summaries and the AlphaQubit source audit |
| `alphaqubit-runtime/` | the runtime lock the freeze and package tools read |
| `score_config.json` | the frozen scoring configuration |

The reference recipe trains ten specialists, one per geometry/basis group. Each gets 18 h of
pretraining and 2 h of fine-tuning on one GPU, selected on validation. The commands for one
group:

```bash
cd /workspace/baseline/alphaqubit
python -m baselines.alphaqubit_train --data-root /workspace/data --group d5_bX_center_5_5 \
  --stage pretrain --hours 18 --dtype bfloat16 --output /workspace/research/models/d5_bX_center_5_5/pretrain
python -m baselines.alphaqubit_train --data-root /workspace/data --group d5_bX_center_5_5 \
  --stage finetune --hours 2 --dtype bfloat16 \
  --init-model /workspace/research/models/d5_bX_center_5_5/pretrain/best.npz \
  --output /workspace/research/models/d5_bX_center_5_5/finetune
```

After all ten groups (`reproduction/groups.txt`), freeze, then package straight into the submission.
`qecbench.package` assembles next to `--output` and swaps the directory in with one rename:

```bash
python -m baselines.alphaqubit_artifact --models-root /workspace/research/models --stage finetune --output /workspace/research/frozen
python -m qecbench.package --package /workspace/research/frozen --output /workspace/submission
```

Scale the hours down for smoke runs. `RESULTS.md` has the reproduced numbers.

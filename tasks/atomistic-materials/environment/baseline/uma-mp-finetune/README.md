# Baseline: UMA-S-1.1 fine-tuned on MP-compatible data (the original task's recipe)

The original task's standard recipe, unchanged. It adapts the UMA-S-1.1 backbone with a newly
initialized conservative E/F/S head on the 500k MPtrj + sAlex sample:
- AdamW 4e-4 with cosine decay;
- loss weights 20/2/1 for energy/forces/stress;
- seed 41, FP32, no EMA;
- global batch 32 on eight GPUs, 3 epochs = 46,875 updates.

The original run took 2 h 16 min on eight H100s (`docs/VALIDATION-v1.md` in the task repository).
Its exported checkpoint is at `/workspace/data/models/mp-finetune-3ep/` (`model.pt` +
`calculator.py`). `RESULTS.md` has its scores on every v2 suite.

## Reproduce

The sample is in `/workspace/data/baselines/mp-standard` (frames touching a v2 evaluation material
removed; see `data/README.md`). Run from `/workspace/baseline/uma-mp-finetune` with the image's
`/opt/venv` Python:

```bash
python prepare_training.py --data /workspace/data/baselines/mp-standard \
  --output /workspace/research/mp-ft --checkpoint /workspace/data/models/uma-s-1p1.pt \
  --epochs 3 --world-size 8 --batch-size 4 --evaluate-every 1000
PYTHONPATH=$PWD fairchem -c /workspace/research/mp-ft/train.yaml
```

Checkpoints are saved every 5,000 steps plus the final step (12 retained, about 26 GB). The original run
chose the minimum labeled-development loss, which was the final checkpoint. Export it as a
submission, using FAIRChem's actual timestamped run directory:

```bash
python export_model.py --checkpoint /workspace/research/mp-ft/runs/TIMESTAMP/checkpoints/final/inference_ckpt.pt \
  --data /workspace/data/baselines/mp-standard --output /workspace/research/mp-ft/submission --weights model
cp calculator.py /workspace/research/mp-ft/submission/
```

`export_model.py` copies the trained weights into FAIRChem's inference slot (no averaging). It also
writes the training provenance (`energy-audit.json`, `training-manifest.json`). Then score every
validation suite with the commands in `/workspace/contract.md`, and swap the directory into
`/workspace/submission`.

`check_consistency.py` checks energy/force/stress finite-difference consistency on labeled frames.
`label_metrics.py` gives E/F/S errors on `val.jsonl`. `plot_learning_curves.py` plots the per-rank
records under the run's `timing/` directory.

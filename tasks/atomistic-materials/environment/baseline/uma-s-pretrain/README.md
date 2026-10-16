# UMA-S pretraining starter

- Random UMA-S backbone and conservative E/F/S head: 146.6M trainable parameters, four layers,
  32 experts, one `omat` domain. No pretrained weights, external fitted offsets or teacher labels.
- Train-only minimum-norm elemental energy fit and force RMS; provenance in `preparation.json`.
- AdamW 4e-4, cosine decay, E/F/S weights 20/2/1, seed 41, FP32, global batch 32, no EMA in epoch mode.
- This starter uses the **498,841 train / 4,995 labeled-dev** v2-filtered MP-compatible sample.
  The task makes all **42,409,826 frames** available. Choosing more frames and combining the
  different DFT settings is part of the task; do not concatenate incompatible total energies blindly.

Inside the agent image, using the existing shared FAIRChem callbacks:

```bash
cd /workspace
python baseline/uma-s-pretrain/prepare_training.py \
  --data data/baselines/mp-standard --output research/pretrain \
  --epochs 3 --world-size 8 --batch-size 4 --evaluate-every 1000
PYTHONPATH=/workspace/baseline/uma-s-pretrain:/workspace/baseline/uma-mp-finetune \
  fairchem -c research/pretrain/train.yaml
```

Three epochs gives 46,767 updates for this sample; this is a smoke/pilot duration, not a convergence
claim. Use `--epochs 30` for 467,670 updates, or a fixed `--steps` budget. The generated receipt
records the actual count. Epoch mode saves every 5,000 updates plus the final checkpoint and keeps
12 checkpoints. Select by labeled-dev loss, never test scores.

```bash
python baseline/uma-s-pretrain/export_model.py \
  --checkpoint research/pretrain/runs/TIMESTAMP/checkpoints/final/inference_ckpt.pt \
  --data data/baselines/mp-standard --preparation research/pretrain/preparation.json \
  --output research/candidate --weights model
```

The export contains `calculator.py`, `model.pt`, initialization/statistics receipts and a README
stub. Complete its compute, data-selection and validation notes, evaluate it using the eight-suite
commands in `/workspace/contract.md`, then publish the directory at `/workspace/submission`.
The same ASE calculator serves all suites; no suite-specific weights or label lookup.

Shared utilities in `../uma-mp-finetune/` include gradient/timing callbacks, learning-curve plots,
and force/stress finite-difference checks. Only these utilities are reused, not that baseline's
checkpoint loader or fitted statistics. This is crystal pretraining with a conservative head from
step zero, not a reproduction of UMA's multi-domain training schedule.

The original 500k/5k, 30-epoch pilot was stopped before physical validation. A separate 10-epoch
trial on 1,000,000 mixed training frames completed eight-suite validation with Score v2 37.08/100.
Its metrics, recipe, training curve and costs are in `RESULTS.md` and `reproduction/`. The commands
above use the smaller MP-compatible starter sample and do not reproduce that trial as written.

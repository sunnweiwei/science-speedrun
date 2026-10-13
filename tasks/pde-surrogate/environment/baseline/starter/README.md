# Baseline: starter (channel-shared U-Net)

A deliberately small, architecture-neutral model that exercises the whole loop in this workspace:
train, publish, replay. **It is not a target to beat, and it is not where a serious attempt
should start.** It reaches reward 0.3127, just above persistence (0.2827) and the spatial-mean
predictor (0.3105) (`RESULTS.md`). Its job is to prove the contract end to end and to give you
something that already trains and publishes while you build the real model.

**Where to start instead.** `../cnext_unet/` runs The Well's CNextU-net recipe: a real
18.6M-parameter architecture trained from scratch, the best measured numbers on PDEArena and the
turbulent layer. `/workspace/references/SOTA_AUDIT.md` covers the rest of the field.

- `model.py`: channel-shared residual U-Net. It takes 4 context frames of one channel and outputs
  the next frame of that channel. Shared weights across channels make it independent of each
  dataset's channel count and grid size.
- `data.py`: training windows from the split manifest's `train` pools, with per-dataset
  per-channel normalization statistics. A missing pool is an error; `--allow-missing-pools`
  records it instead of hiding it.
- `train.py`: per-dataset mini-batches, random 128² crops, MSE in normalized space.
  - `--datasets` restricts training to a subset.
  - `--max-windows-per-traj` bounds indexing cost for a smoke run.
  - `--seed` fixes initialization and sampling.
  - It saves the last step only. There is no validation loop and no checkpoint selection;
    `evaluation/make_valid_cases.py` plus `evaluation/score.py` is the selection loop to add.
- `predict.py` / `infer.sh`: the contract entry point. Unseen datasets are normalized with the
  statistics of their own context frames.
- `publish.py RUN_DIR SUBMISSION_DIR`: assembles the portable package in a staging directory and
  swaps it into `SUBMISSION_DIR` with one rename.

```bash
cd /workspace/baseline/starter
# smoke: one small dataset, a few minutes
python3 train.py --data /workspace/data --manifest /workspace/data/split.json \
  --datasets well/turbulent_radiative_layer_2D --max-windows-per-traj 4 \
  --out /workspace/research/starter-smoke --steps 300 --batch 4 --width 16
# the recorded run (RESULTS.md): 30,000 steps, 24 minutes on one H100
python3 train.py --data /workspace/data --manifest /workspace/data/split.json \
  --out /workspace/research/starter --steps 30000 --batch 8 --width 32 --crop 128 --lr 1e-3 --seed 0
python3 publish.py /workspace/research/starter /workspace/submission
```

Known weaknesses you can address:

- no cross-channel coupling;
- no PDE or parameter conditioning;
- zero padding on periodic domains;
- no long-rollout training objective;
- no re-implementation of the stronger published architectures.

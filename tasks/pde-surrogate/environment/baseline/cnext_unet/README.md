# Baseline: The Well's CNextU-net, trained from scratch

The Well's published baseline checkpoints carry no license. So this baseline retrains the best of
them (CNextU-net, "outperforms the others on 8 of 17 experiments") from the BSD-3 `the_well`
package with the published recipe:

- `UNetConvNext(init_features=42, blocks_per_stage=2)`, 18.6M parameters;
- AdamW, lr 1e-2, weight decay 1e-4, cosine schedule with 5 warm-up epochs;
- batch 64, four input frames, z-score normalization, MSE loss;
- 200 epochs (the paper capped at 500 epochs or 12 h on one H100).

It is one model per dataset, as in the paper, so it is a per-dataset reference rather than a
cross-dataset model. Its package serves PDEArena Navier-Stokes. Other 3-channel datasets also
pass through the model, and datasets with another channel count get persistence. It reaches
reward 0.3447, the best measured VRMSE on PDEArena (`RESULTS.md`).

Everything it needs is in this image:

- The Well's source is at `/opt/the_well`, commit `ad50de0e8861380d7c4d62c0503c654710f026e7`.
  Its Hydra configs are not in the wheel.
- `hydra-core`, `wandb` (offline) and `torchinfo` are installed.

The scripts read `/workspace/data` and write under `/workspace/research/cnext_unet`.

```bash
cd /workspace/baseline/cnext_unet
# PDEArena Navier-Stokes: write the task's train/valid pools in The Well's layout (a few minutes)
python3 convert_pdearena_to_well.py --data /workspace/data --manifest /workspace/data/split.json \
  --out /workspace/research/cnext_unet/well-pdearena_navier_stokes_2d
# train from scratch (RESULTS.md: 200 epochs, about 10 h on one H100); the turbulent-layer data config
# is reused, as in the recorded run
WELL_EXTRA_DATASETS=pdearena_navier_stokes_2d \
  bash train_dataset.sh pdearena_navier_stokes_2d turbulent_radiative_layer_2D 200
# wrap the best-validation checkpoint as a contract package and publish it (one atomic swap)
python3 package.py --run /workspace/research/cnext_unet/runs/<run directory> \
  --dataset-dir /workspace/research/cnext_unet/well_base/pdearena_navier_stokes_2d \
  --dataset-name pdearena/navier_stokes_2d --checkpoint best.pt --out /workspace/submission
```

`train_trl2d.sh` is the same recipe on The Well's turbulent radiative layer
(`/workspace/data/well-turbulent_radiative_layer_2D`); pass
`--dataset-name well/turbulent_radiative_layer_2D` to `package.py` for that model.

- `train_dataset.sh` links the read-only datasets into a writable base under the research
  directory, because The Well's loader wants `<base>/<dataset>/data/{train,valid,test}`.
- The workspace has no test split: the held-out cases stay with the verifier. So `test` points at
  `valid`, and the trainer's closing test pass reports validation numbers. The task score comes
  only from the independent replay.
- `register_local_dataset.py` adds the converted dataset's name to The Well's list of known
  datasets; nothing else about the recipe changes.
- The published batch size fits in 40 GB only with `+model.gradient_checkpointing=true`. This
  changes memory and speed, not the optimization.

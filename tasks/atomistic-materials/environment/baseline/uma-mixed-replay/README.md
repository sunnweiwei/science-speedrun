# Baseline: MP fine-tune with OMat24 off-equilibrium replay

The MP fine-tune recipe of `../uma-mp-finetune` with the same optimiser, step budget (46,875
updates at global batch 32 on eight GPUs), new conservative E/F/S head and seed. Half of every batch
comes from OMat24 instead:

- `omat`: 500,000 MPtrj + sAlex frames from `/workspace/data/pool`. Training targets: energy,
  forces and stress (MP-compatible raw energies, MPA elemental references).
- `offeq`: 500,000 OMat24 frames (rattled, rattled-relax and AIMD subsets). Training targets:
  forces and stress only, because OMat24 energies are on a different DFT scale. The backbone maps
  this config name to the pretrained `omat` dataset embedding.

Frames are chosen by SHA-256 priority of their IDs, with at most four frames per parent material.
A composition-disjoint development split of 2,500 + 2,500 frames is held out from fitting. Each
frame is seen 1.5 times.

## Reproduce

From `/workspace/baseline/uma-mixed-replay`, with `/opt/venv` Python on eight GPUs:

```bash
python prepare_mixed.py --pool /workspace/data/pool --checkpoint /workspace/data/models/uma-s-1p1.pt \
  --output /workspace/research/mixed
PYTHONPATH=/workspace/baseline/uma-mp-finetune:/workspace/baseline/uma-mixed-replay fairchem -c /workspace/research/mixed/train.yaml
python export_submission.py \
  --checkpoint /workspace/research/mixed/runs/TIMESTAMP/checkpoints/final/inference_ckpt.pt \
  --output /workspace/research/mixed/submission
```

`prepare_mixed.py` takes about 16 minutes (96 workers); training took 2 h 17 min on eight H100s (2 h 2 min of step compute; the final checkpoint had the lowest development loss). Score the
exported directory on every validation suite with the commands in `/workspace/contract.md`. The trained
weights are at `/workspace/data/models/mixed-replay/`, and `RESULTS.md` has the scores.

This baseline is a reference point, not a tuned method. Open questions are left to the agent:
- the OMat24 fraction and subsets, and MatPES;
- whether to reference OMat24 energies to the MP scale instead of dropping them;
- schedules, and how to keep the MP energy scale.

# Baseline: starter-41m (the starter recipe at 41.8M parameters, eight GPUs)

A reference baseline (the main one is `../timesfm3-50m`). It uses the code in `../starter`, at 6.5× the parameters and about 70× the
training steps of the small starter:
- model: d_model 512, 12 layers, 8 heads, 41.8M parameters;
- training: 2,000,000 steps, global batch 256, AdamW at 4e-4, 5,000 warm-up steps, cosine decay
  over the whole run, 10 % KernelSynth synthetic series;
- checkpoint: the one with the best validation skill (evaluated every 10,000 steps).

This is the recipe of the task author's single-GPU calibration run (28 GPU-hours on an A100-40GB,
`docs/validation/task_starter_scaled_41m.json`). Here it is trained data-parallel on eight GPUs:
- each rank takes 32 windows of the 256-window global batch;
- gradients are averaged across ranks;
- steps, learning rate and schedule are unchanged.

```bash
cd /workspace/baseline/starter
python3 data.py pack --data /workspace/data/gifteval_pretrain --out /workspace/research/packed
torchrun --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 29511 --nproc_per_node 8 train.py \
  --packed /workspace/research/packed --out /workspace/research/runs/41m --steps 2000000 --batch 256 \
  --lr 4e-4 --warmup 5000 --d-model 512 --layers 12 --heads 8 --context 1024 --horizon 128 --patch 32 \
  --synthetic-fraction 0.1 --valid /workspace/data/valid --eval-every 10000 --seed 0
python3 publish.py --weights /workspace/research/runs/41m/best.pt --out /workspace/research/submission-41m \
  --sources-from /workspace/research/packed --synthetic kernelsynth
```

`--master_addr 127.0.0.1` is needed because the workspace has no network: torchrun's default
rendezvous resolves the container hostname, which times out.

Validate and publish as in `../starter/README.md`. `RESULTS.md` has the reproduced score and times.
`../starter` (6.4M, 30,000 steps, minutes on one GPU) remains as a quick reference for the full
pipeline.

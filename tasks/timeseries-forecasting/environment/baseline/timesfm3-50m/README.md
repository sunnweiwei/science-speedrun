# Baseline: timesfm3-50m (the TimesFM-3 architecture from scratch, eight GPUs)

The task's main baseline: TimesFM-3's architecture, scaled down to 50.9M parameters and trained
from random initialization on the task corpus.

- **Model.** `TimesFM3Torch` from TimesFM-3's released PyTorch code (`vendor/timesfm3`,
  google-research/timesfm 3.0.2, Apache-2.0; code only, no weights):
  - d_model 768, 14 layers, 12 heads;
  - input patch 32, output patch 64, nine quantiles;
  - RevIN, rotary positions, no variate attention (univariate).
- **Objective.** Every input patch forecasts the next 64 points. The loss is pinball loss on the
  nine quantiles in the patch's running-normalized units. Contexts have random length (left-masked).
- **Data mixture** (`mixture.py`). Windows are drawn by source group, then by directory with
  weight ∝ sqrt(points), then uniformly by series:

  | group | sources | share |
  | --- | --- | ---: |
  | diverse | the original 81 directories | 50 % |
  | climate | ERA5, CMIP6 | 25 % |
  | buildings | buildings_900k | 15 % |
  | cloud | borg_cluster_data_2011 | 10 % |

  10 % of all windows are KernelSynth synthetic series. Uniform-over-series sampling (`--sampling
  series`) would draw 88 % of windows from climate and buildings data.
- **Optimization.** Global batch 1024 over eight ranks, AdamW 5e-4, 2,000 warm-up steps, cosine
  decay over 150,000 steps, bf16.
- **Checkpoint selection.** Every 5,000 steps rank 0 forecasts 15 stratified validation configs with
  the released TimesFM-3 forecaster and scores them with the task grader; `best/` is the best
  validation skill.
- **Inference** (`predict.py`). The released `TimesFM3Evaluator` with the GIFT-Eval notebook flags,
  run univariate, with the context capped at the training context (2,048).

```bash
cd /workspace/baseline/starter
python3 data.py pack --data /workspace/data/gifteval_pretrain --out /workspace/research/packed   # ~20 min, 236 GB
cd /workspace/baseline/timesfm3-50m
torchrun --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 29531 --nproc_per_node 8 train_tfm3.py \
  --packed /workspace/research/packed --out /workspace/research/runs/tfm3 --steps 150000 --batch 1024 \
  --lr 5e-4 --warmup 2000 --context 2048 --dims 768 --layers 14 --heads 12 --synthetic-fraction 0.1 \
  --sampling mixture --eval-every 5000 --seed 0
python3 publish.py --checkpoint /workspace/research/runs/tfm3/best --out /workspace/research/submission-tfm3 \
  --sources-from /workspace/research/packed
bash /workspace/research/submission-tfm3/infer.sh INPUT_DIR OUTPUT_DIR
```

`--master_addr 127.0.0.1` is needed because the workspace has no network. Validate and swap the
package into `/workspace/submission` as in `../starter/README.md`. `RESULTS.md` has the score and times.

Measured while designing this baseline (`docs/validation/corpus_scaling.json` in the task
repository): with uniform sampling, the same model overfit the 41 GB v1 corpus within about
25,000 steps and did better on the 273 GB v2 corpus. Directions this baseline leaves open:
- the mixture weights;
- longer contexts (TimesFM-3 itself uses up to 15,360);
- variate attention for the multivariate configs;
- model size;
- the long-horizon behaviour that dominates the reward.

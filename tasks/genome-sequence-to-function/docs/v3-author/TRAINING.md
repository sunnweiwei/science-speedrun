# Compact materialization and training

The frozen and only supported profile contains 3,200 train, 200 validation,
and 1,000 test examples. Every selected example retains all 5,930 human tracks
in eleven output families.

## Start materialization on `download`

After synchronizing the branch to
`/home/zifeng/ai4sci-tasks/genome-sequence-to-function`, run:

```bash
cd /home/zifeng/ai4sci-tasks/genome-sequence-to-function
nohup bash environment/data/materialize.sh release \
  /home/zifeng/agbench-4400 \
  > /home/zifeng/agbench-4400.log 2>&1 &
echo $! > /home/zifeng/agbench-4400.pid
```

Monitor without attaching to the data process:

```bash
tail -f /home/zifeng/agbench-4400.log
```

The job is source-shard resumable. Re-running the same `release` command reads
validated index/Zarr checkpoints and resumes at the next incomplete shard.
Upstream TFRecords are streamed from GCS and are not retained locally. The
worker should have at least 120 GiB free to leave room for Docker and temporary
shards even though the projected final package is about 91.8 GiB.

## Model adapter

The track uses one `CompactModel` and reader:

- 1,024-bp transformer trunk (1,024 tokens for a 1-Mb input);
- 128-bp heads for the seven dense families;
- base-resolution splice-site logits;
- event-query heads for splice usage and junctions;
- a symmetric 2-kb contact-map head;
- equal family weighting during training;
- streaming per-track Pearson and mean-per-family reporting during evaluation.

The checked-in configuration has 13,218,756 trainable parameters. A real
1-Mb example has passed forward and backward across all eleven families on the
remote worker.

Junction evaluation is explicitly reported on stored positive events. It is a
baseline diagnostic and is not presented as the official dense all-pair
junction metric.

Continuous targets are trained and scored in `asinh` space, which is defined
for signed contact values as well as nonnegative count tracks. Every loss and
metric respects its stored validity mask. Empty event sets contribute zero
loss rather than `NaN`.

Build the existing starter image on a GPU worker, then train and evaluate:

```bash
cd /path/to/genome-sequence-to-function
docker build -t alphagenome-50m-dev environment
bash environment/starter/reference/alphagenome_50m/run_pipeline.sh \
  /path/to/agbench-4400 /path/to/run 10 0
```

The wrapper requires a complete validation receipt, trains for ten epochs
by default, resumes from the last completed epoch in `last.pt`, selects the
best checkpoint on validation, and writes `valid_metrics.json`. A periodic
`progress.pt` is diagnostic only; resuming it would repeat part of an epoch,
so training rejects it. Formal test inference and trusted grading use the
selected `best.pt` separately. Materialization is suitable for the CPU
`download-genome` instance; full model training requires a CUDA GPU worker.

## Materialized-data tiny overfit check

The checked-in [tiny overfit configuration](environment/starter/reference/alphagenome_50m/tiny_overfit.yaml)
keeps the frozen 1-Mb input, target geometry, and all eleven output families,
but shrinks the model to 226,248 parameters. It trains on the first materialized
training example for 120 steps. Run from this directory with the existing
`alphagenome-50m-dev:latest` image and an NVIDIA GPU:

```bash
bash environment/starter/reference/alphagenome_50m/run_tiny_overfit.sh \
  /mnt/c/github/datasets/genome-sequence-to-function \
  /mnt/c/github/rsi-for-science/ai4sci-tasks-alphagenome/runs/alphagenome-tiny-overfit-120
```

The 2026-09-30 run used seed 17, AdamW with constant learning rate 0.003 and
zero weight decay. The same-sample supervised loss fell from **5.3842** to
**0.8150** (84.86% reduction). The per-family losses, example ID, dataset
release hash, and checkpoint hash are saved in
[the experiment report](results/tiny_overfit_2026-09-30.json). The local
`runs/alphagenome-tiny-overfit-120/` directory retains `last.pt`, the training
log, and the generated JSON report. The report measures the training objective
with supplied splice coordinates on one training example; it is an overfit
diagnostic, not a validation or benchmark score.

## Three-epoch tiny model on the full release

The full-data runner uses the same tiny architecture and all 3,200 materialized
training examples. It trains for three epochs with AdamW, batch size one,
learning rate 0.001 decaying to 0.0001, and weight decay 0.0001. It evaluates
all 200 validation examples after each epoch, retains the checkpoint with the
highest equal mean of eleven family Pearson scores, and evaluates all 1,000
held-out test examples once after checkpoint selection:

```bash
bash environment/starter/reference/alphagenome_50m/run_tiny_full.sh \
  /mnt/c/github/datasets/genome-sequence-to-function \
  /mnt/c/github/rsi-for-science/ai4sci-tasks-alphagenome/runs/alphagenome-tiny-full-3e-2026-09-30 \
  3
```

The test evaluator uses predicted splice positions and the same stored hidden
coordinate queries as the grader. It runs FP32 inference and rounds model
outputs to FP16 before computing per-track Pearson. This is a local trusted
held-out evaluation, not a sealed inference and grader replay. The latter
requires writing the full sharded test prediction package.

The 2026-09-30 run completed all 9,600 training steps. The selected epoch-3
checkpoint has SHA-256
`2fb135e66c57d258637e9c989b84f7bd75db0942528cac089161ad0f2277d360`.
The comparable validation scores were **0.09462** after epoch 2 and **0.12863**
after epoch 3; local held-out test mean-family Pearson was **0.14236**. The
epoch-1 validation score, 0.00027, was produced before the evaluator's FP32 /
FP16 and float64 moment correction, so it is retained only as a diagnostic.
The [result summary](results/tiny_full_3epoch_summary_2026-09-30.json) links
the full per-track validation and test reports. The local `runs/` directory
retains the checkpoint and logs.

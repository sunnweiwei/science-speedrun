# temporal-gru: the verified baseline

`longitudinal.py` is a complete neural workflow for this task. It fits train-only numeric
statistics and the event vocabulary, builds keyed caches, trains one shared model for all
diseases and death, selects a checkpoint on a train-internal fold, calibrates on a second disjoint
train fold, and writes the prediction parquet. Measured scores are in [RESULTS.md](RESULTS.md).

Three modes share the cache:

- `temporal` (the baseline): multimodal event embeddings, numeric value/missingness channels,
  time features, a GRU, and learned cumulative interval hazards for all diseases and death;
- `unordered`: the same history without ordered recurrence;
- `diagnosis`: a diagnosis-only temporal ablation.

None of these choices is a task constraint. The deterministic 512-event sample, the 64-dimensional
embeddings, the 128-dimensional GRU, mean pooling, the loss weights, the optimizer and the hash
folds are all open to change.

## Train, score and publish

About 4 min of caching, 4–7 min of training on one GPU and under a minute to predict:

```bash
B=/workspace/baseline/temporal-gru; W=/workspace/research/gru
python $B/longitudinal.py cache --input /workspace/data/train \
  --labels /workspace/data/train/label_windows.parquet --cache $W/train.pt
python $B/longitudinal.py train --materialized /workspace/data --cache $W/train.pt \
  --kind temporal --epochs 15 --batch-size 64 --seed 20260929 --device cuda --output $W/model.pt
python $B/longitudinal.py calibrate --cache $W/train.pt --checkpoint $W/model.pt \
  --device cuda --output $W/model-calibrated.pt

# package, score on validation, publish
bash $B/package.sh $W/model-calibrated.pt $W/train.adapter.json $W/package
bash $W/package/infer.sh /workspace/data/public/validation $W/validation
python /workspace/evaluation/score.py --input /workspace/data/public/validation \
  --trusted /workspace/data/trusted/validation \
  --normalization /workspace/data/trusted/risk_normalization.parquet \
  --predictions $W/validation/predictions.parquet --output $W/validation-score
python $B/publish.py $W/package --check
```

`score.py` writes `report.json`, with `panel` holding the reward and every endpoint, and
`reward.json`. `package.sh` copies the model code, weights and train-fit adapter, and writes an
`infer.sh` that rebuilds the input cache offline. `publish.py` swaps the package into
`/workspace/submission` atomically; with `--check` it first runs `infer.sh` on the validation
inputs.

Set `OMP_NUM_THREADS` to the CPUs you have (the image sets 10). Without it, PyTorch sizes its
thread pool by the host's cores, and under a CPU quota training ran about 15× slower.

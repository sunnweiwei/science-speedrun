# alphagenome-reference: the verified baseline

An AlphaGenome-style reference model with its complete training, inference and packaging code,
trained on the full train split with every GPU. Its architecture follows AlphaGenome:
- a convolutional DNA encoder from 1 bp down to 1,024-bp tokens;
- a transformer over the 1,024 tokens;
- a decoder back to 128-bp bins and base resolution;
- heads for the seven dense families, splice sites, splice usage, junctions (low-rank
  donor × acceptor) and contact maps.

The baseline is the 215M configuration (`reference_large.yaml`); its scores are in
[RESULTS.md](RESULTS.md). The 13.2M default configuration was not trained at full scale.

| config | parameters | width / depth | single-H100 step |
|---|---:|---|---:|
| `alphagenome_50m.yaml` | 13.2M | the code's default | 0.05 s |
| `reference_large.yaml` | 215.3M | encoder 96 (+96 per level), trunk 1,024 × 16 blocks, 16 heads, head 768 | 0.10 s |

None of these choices is a task constraint: architecture, scale, objective, data use and schedule
are open.

## Files

| file | role |
|---|---|
| `reference/alphagenome_50m/` | model (`model.py`, `layers.py`), data loader with 8-bit target decoding (`dataset.py`), training and streaming validation (`train_eval.py`), configurations, unit tests |
| `reference/alphagenome_50m/run_ddp.sh` | multi-GPU training on the full train split, then validation scoring |
| `predict.py`, `infer.sh` | track predictions (`predictions.zarr`) with sparse splice usage and junctions |
| `perturb.py`, `perturb.sh` | A/B outputs (float32) for every row of a perturbation table |
| `package.sh` | a self-contained package from a checkpoint: this code plus `last.pt` |
| `publish.sh` | atomic swap of a package into `/workspace/submission` |

The predictor reads only DNA and the public manifests. It picks splice sites from its own
splice-site logits, writes usage at up to 4,096 of them, and junctions for the highest-scoring
16,384 (strand, donor, acceptor) pairs.

## Train, score and publish

One example per GPU per step; the cosine schedule spans the requested epochs. With eight GPUs the
large configuration runs at about 0.21 s per step, about 18 min per epoch over 41,020 intervals:

```bash
B=/workspace/baseline/alphagenome-reference; W=/workspace/research/large
cd $B
# train (8 GPUs, 12 loader workers each), then score validation with the verifier's scorer
bash reference/alphagenome_50m/run_ddp.sh /workspace/data $W reference/alphagenome_50m/reference_large.yaml 14 8 12
cat $W/valid-score/report.json

# development variant split: answer its queries and score them
python $B/perturb.py /workspace/data $W/valid-variants --checkpoint $W/last.pt \
  --queries /workspace/data/variant_evals/valid/public/perturbations.parquet
python /workspace/evaluation/score_variants.py --perturbations $W/valid-variants/perturbations.zarr \
  --labels /workspace/data/variant_evals/valid --output $W/valid-variant-score

# package and publish atomically
bash $B/package.sh $W/last.pt $W/package
bash $B/publish.sh $W/package /workspace/submission
```

`run_ddp.sh` writes `progress.pt` every 2,000 steps; pass `--resume` to `train_eval.py` to continue
after an interruption. Unit tests: `cd $B && python -m pytest reference/alphagenome_50m/tests`.

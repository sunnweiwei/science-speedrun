# Small training reference

This package is a source snapshot from `sunnweiwei/protein-folding` commit
`09900626ae9335df610cddbc0801a2acdc0a7bd4`. Its feature adapter and input width
are adapted to the task's canonical ESMC-6B final-hidden INT8 format; the rest
remains a small reference rather than a prescribed architecture.

These models are compact **plumbing/smoke references**, not architecture-faithful
ESMFold/Protenix reproductions and not trained FoldBench-ready solutions. They
demonstrate OpenFold3 NPZ loading, ESMC-6B final-hidden INT8 features, crops, DDP training,
losses and checkpoint saving. Complete all-atom inference still needs development.

Run from `/workspace/baseline/starter` (or put that directory on PYTHONPATH):

```bash
python -m reference.build_reference_index \
  --manifest /workspace/data/manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl \
  --output-dir /workspace/research/my-index --data-root /workspace/data

python -m reference.train_reference \
  --config reference/configs/pairformer_lite_50m.json \
  --train /workspace/my-index/train.jsonl \
  --validation /workspace/my-index/validation.jsonl \
  --output-dir /workspace/my-training --steps 20
```

The indexer's split is merely an optional example; all 100K records remain
available and the agent chooses its own split. It records no MSA paths and the
supplied reference models do not consume MSA. The task's no-MSA rule applies.
The 50M example size, crop sizes, schedules and model
families are starting points, not benchmark restrictions.

# Atom-histogram starter

The starter counts atom types in each molecule and binding pocket, embeds
the two count vectors, and learns to rank paired structures above mismatches.
Use it for a small end-to-end test of training, export, and publication.

From a fresh offline development workspace:

```bash
python /workspace/references/minimal-starter/train.py \
  --input-root /workspace/data --output-dir /workspace/research/minimal \
  --epochs 1 --batch-size 4 --workers 0 \
  --max-pretrain-records 8 --max-train-records 8 --max-validation-records 8
python /workspace/publish.py /workspace/research/minimal
```

Use a fresh output directory for each experiment. The controller can replay
the published package in the independent verifier. The scientific reference
and its recorded results are in [DrugCLIP](../../baseline/drugclip/README.md).

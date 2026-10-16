# Data

AlphaGenome's public human training data, `FOLD_0` split: every eligible interval of the training
folds, plus the validation fold. Everything under `/workspace/data/` except this file is mounted
read-only.

| path | content |
|---|---|
| `reference/GRCh38.p13.genome.fa` (+ `.fai`) | the pinned reference genome; all sequence comes from here |
| `public/{train,valid}.parquet` | manifests: `example_id`, chromosome, 1,048,576-bp input window with padding, and the central 196,608-bp target window. The test manifest, same format, stays with the verifier |
| `public/junction_tracks.parquet` | the 734 splice-junction tracks and their strand half |
| `metadata/tracks.parquet` | all 5,930 tracks: family, assay, biosample and ontology term, strand, data source |
| `labels/{train,valid}.parquet` | per example: where its targets are (`target_shard`, `target_group`) and its splice-site positions |
| `targets/train/` | 41,020 examples in 250 Zarr shards, 8-bit (see below), 331 GB |
| `targets/valid/` | 200 examples, full precision |
| `scoring/{train,valid}/` | junction scoring queries per example: the (donor, acceptor) pairs scored for junctions |
| `variant_evals/valid/public/perturbations.parquet` | the development split's variant and perturbation queries, in the format `perturb.sh` answers; the 6,587 test queries and their labels stay with the verifier |
| `variant_evals/valid/` | the development split of the same 13 evaluations: 2,697 queries (`public/`) with labels and published anchors (`labels/`) |

## Targets

Per example and family, matching the output contract (`/workspace/contract.md`):
- dense families (`atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone`):
  `[1536, tracks]` 128-bp bins of the target window in `asinh(count)`, with a validity mask;
- `splice_sites` `[196608, 4]`, `splice_usage` and `junctions` at observed sites and pairs, and
  `contact_maps` `[131328, 28]` (2,048-bp upper triangle).

**Train targets are quantized.**
- Dense values are stored as `uint8` codes of `asinh(x)` between per-track `values_lo` and
  `values_hi` (attribute `encoding = "asinh-uint8/v1"`).
- Contact maps are coded linearly (`linear-uint8/v1`).
- `baseline/alphagenome-reference/reference/alphagenome_50m/dataset.py` (`_decode`) restores both per example in the
  data loader.

The round trip keeps per-track Pearson ≥ 0.9997 in `asinh` space. Training the reference on
quantized and on full-precision targets gave the same validation score within seed noise.
Validation targets, and the hidden test targets, are full precision.

## Splits

The split is AlphaGenome's `FOLD_0`:
- train on folds 2–7;
- validate on fold 0 (a fixed sample of 200 intervals);
- the hidden test is fold 1.

The 674 training intervals whose input overlaps a validation or test target window are excluded.

## Variant queries

Each row of `perturbations.parquet` asks for the model's outputs on two sequences of one
1,048,576-bp window:
- an SNV with two explicit alleles; or
- the reference against an N-masked interval.

The row names the families needed and the bins or bases to return. The test labels stay with the
verifier. `/workspace/evaluation/variant_scoring.py` shows exactly how A/B outputs become variant
scores.

**Development split.** `variant_evals/valid/` holds the same 13 evaluations at about 40 % of the
test size, with labels. Its variants never appear in the test split. AlphaGenome evaluations come
from the validation fold (fold0) where it has enough positives; MFASS and bQTL come from loci
outside the test fold. TraitGym uses match groups that the test split does not. Score it with:

```bash
python /workspace/baseline/alphagenome-reference/perturb.py /workspace/data OUT --checkpoint last.pt \
  --queries /workspace/data/variant_evals/valid/public/perturbations.parquet
python /workspace/evaluation/score_variants.py --perturbations OUT/perturbations.zarr \
  --labels /workspace/data/variant_evals/valid --output OUT/score
```

The queries come from:
- AlphaGenome's evaluations: eQTL, sQTL, ClinVar, MFASS, caQTL, bQTL and CRISPRi enhancer–gene;
- TraitGym.

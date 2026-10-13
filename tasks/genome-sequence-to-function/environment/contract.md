# Benchmark contract

A submission is a directory with two executables. Both receive the same read-only `INPUT_ROOT`,
which holds the pinned reference FASTA (`reference/GRCh38.p13.genome.fa`) and public manifests
only. Neither receives labels, scoring queries, splice coordinates or targets. Both run offline as
an unprivileged user on one H100 (80 GB) and share one time budget of 4 hours. The 215M-parameter
baseline needs about 20 minutes of it (10 for `infer.sh`, 10 for `perturb.sh`).

## 1. Track prediction: `infer.sh INPUT_ROOT OUTPUT_ROOT`

Input: `public/test.parquet`, giving sample identity, chromosome, 1,048,576-bp input coordinates,
padding and the central 196,608-bp target window.

Output: manifest-order `example_ids.json` and `predictions.zarr`, schema
`genome-sequence-to-function-predictions/v4`. Each example group `examples/<example_id>` holds:

| array | shape | meaning |
|---|---|---|
| `atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone` | `[1536, tracks]` | 128-bp bins of the target window, in `asinh(count)` space |
| `splice_sites` | `[196608, 4]` | per-base scores: donor+, acceptor+, donor−, acceptor− |
| `splice_usage_positions` | `[K]` int32, K ≤ 4,096 | sorted, unique target-window positions of predicted splice sites |
| `splice_usage` | `[K, 734]` | usage at those positions (asinh) |
| `junction_donor`, `junction_acceptor` | `[P]` int32, P ≤ 16,384 | input-window coordinates of predicted junction endpoints |
| `junction_half` | `[P]` int8 | 0 = plus-strand pair, 1 = minus-strand pair; (half, donor, acceptor) is unique |
| `junctions` | `[P, 734]` | junction values for those pairs (asinh) |
| `contact_maps` | `[131328, 28]` | 2,048-bp upper triangle |

Splicing outputs are sparse. Usage at a hidden site the model did not list counts as a zero
prediction. A hidden junction whose (strand, donor, acceptor) pair was not listed also counts as
zero. Both stay in the metric. Predictions must be finite; the tree may not exceed 80 GiB.

**Score.** Pearson per track in `asinh` space over valid targets, averaged within each of the
eleven families, then equally across families: `overall_mean_family_pearson`. Usage is gathered
at hidden positions and junctions at hidden query pairs. This is `reward`.

## 2. Variant and perturbation effects: `perturb.sh INPUT_ROOT OUTPUT_ROOT`

Input: `public/perturbations.parquet`, contract `genome-sequence-to-function-perturbations/v1`.
Each row is a 1,048,576-bp window (`chrom`, `input_start`) and an edit spanning
`[edit_start, edit_end)` in genomic 0-based coordinates:

- `snv`: sequence A carries `allele_a`, sequence B carries `allele_b`. Both alleles are explicit,
  and either may differ from the reference genome.
- `mask`: sequence A is the reference; in sequence B the interval is replaced by N.

Output: `perturbations.zarr`. Its `perturbation_ids` attribute lists the manifest order, and group
`perturbations/p{index:06d}` holds `a/<family>` and `b/<family>` for every family named in
`families`:
- dense families are sliced to bins `[dense_bin_start, dense_bin_end)` of the target window;
- `splice_sites` is sliced to bases `[splice_start, splice_end)` of the target window.

The values are the same quantities as in `predictions.zarr`, as float32. float16 is accepted but
rounds away most single-variant effects (its resolution is about 0.004 at `asinh` values of 4–8).

**Score.** The verifier turns A/B into the published variant scorers and reports each evaluation
next to anchors computed on the same rows. The evaluations are AlphaGenome's eQTL sign, causality
and effect size, sQTL, ClinVar (non-coding, splice region and missense), MFASS splicing, caQTL,
SPI1 bQTL and CRISPRi enhancer–gene linking, plus TraitGym Mendelian and complex-trait causal
variants. Definitions are in `/workspace/evaluation/variant_scoring.py`. A development split with labels
(`/workspace/data/variant_evals/valid`) is scored the same way by `score_variants.py`. These metrics are reported
in `report.json` under `variant_effects`; they are not part of `reward`.

## Rules

- **Model size is not capped.** The training budget bounds it. A single fixed model, or an
  ensemble, serves every output and both entry points.
- Do not use test labels or label-derived caches, and do not use pretrained AlphaGenome parameters
  or any model trained on the evaluation folds.
- Inference is offline. The package must contain everything it needs.

This compact benchmark does not reproduce the full AlphaGenome paper evaluation.

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

**Axis 1 (`tracks_axis`, weight `0.5` in `score` / `reward`).** Pearson $r$ per track in `asinh`
space over valid targets (constant prediction on a varying target = `0.0`), averaged within each of
the eleven output families, then averaged within each of the **5 `TRACK_GROUPS` modalities**:
- `Open chromatin`: `atac`, `dnase`
- `Transcription`: `procap`, `cage`, `rna_seq`
- `Protein binding and histones`: `chip_tf`, `chip_histone`
- `Splicing`: `splice_sites`, `splice_usage`, `junctions`
- `Chromatin contacts`: `contact_maps`

`tracks_axis` is the equal arithmetic mean across those 5 modalities. Usage is gathered at hidden
positions and junctions at hidden query pairs. The ungrouped 11-family mean
`overall_mean_family_pearson` is retained in `report.json` and `reward.json` as a diagnostic.

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

**Axis 2 (`variants_axis`, weight `0.5` in `score` / `reward`).** The verifier turns A/B into the
published variant scorers (`/workspace/evaluation/variant_scoring.py`) across 13 evaluations:
AlphaGenome's `eqtl_sign`, `eqtl_causality`, `eqtl_coefficient`, `sqtl_causality`,
`clinvar_noncoding`, `clinvar_splice_region`, `clinvar_missense`, `mfass`, `caqtl`, `bqtl_spi1`,
and `enhancer_gene`, plus TraitGym's `traitgym_mendelian` and `traitgym_complex`. A development
split with labels (`/workspace/data/variant_evals/valid`) is scored identically by
`score_variants.py` (or `score.py --perturbations`).

To form `variants_axis`, each of the 12 non-redundant evaluations (**explicitly excluding
`eqtl_sign`**, which is retained in `report.json` as a diagnostic because signed Spearman
`eqtl_coefficient` already measures directional effect size) is chance-calibrated as
$v_k = (m_k - b_k) / (1 - b_k)$ using the frozen constant-predictor anchors `GENOME_CONSTANT` in
`/workspace/evaluation/scoring.py`:
- `eqtl_causality`: `0.5`, `eqtl_coefficient`: `0.0`
- `sqtl_causality`: `0.5693860386879731`, `caqtl`: `0.20125786163522014`, `bqtl_spi1`: `0.20040485829959515`
- `enhancer_gene`: `0.2`
- `clinvar_noncoding`: `0.20040080160320642`, `clinvar_splice_region`: `0.20040899795501022`, `clinvar_missense`: `0.20366598778004075`, `mfass`: `0.19936708860759494`
- `traitgym_mendelian`: `0.10182767624020887`, `traitgym_complex`: `0.10012674271229405`

The 12 calibrated scores are aggregated into **5 equal-weight biological branches**:
1. `eQTL` = `mean(v["eqtl_causality"], v["eqtl_coefficient"])`
2. `Other molecular QTL` = `mean(v["sqtl_causality"], v["caqtl"], v["bqtl_spi1"])`
3. `CRISPRi enhancer–gene` = `v["enhancer_gene"]`
4. `Clinical and functional assays` = `0.5 * mean(v["clinvar_noncoding"], v["clinvar_splice_region"], v["clinvar_missense"]) + 0.5 * v["mfass"]`
5. `Human traits` = `mean(v["traitgym_mendelian"], v["traitgym_complex"])`

`variants_axis` is the equal mean of those 5 branches.

**Combined Primary Score (`reward.json["score"]` and `reward.json["reward"]`).**
`signed_score = 0.5 * tracks_axis + 0.5 * variants_axis`, and
`score = reward = clip(signed_score, 0.0, 1.0)`. A trivial constant predictor achieves exact
`tracks_axis = 0.0`, `variants_axis = 0.0`, and `score = 0.0`.

## Rules

- **Model size is not capped.** The training budget bounds it. A single fixed model, or an
  ensemble, serves every output and both entry points.
- Do not use test labels or label-derived caches, and do not use pretrained AlphaGenome parameters
  or any model trained on the evaluation folds.
- Inference is offline. The package must contain everything it needs.

This compact benchmark does not reproduce the full AlphaGenome paper evaluation.

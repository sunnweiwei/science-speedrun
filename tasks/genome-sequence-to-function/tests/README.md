# Verifier

Two services, both offline (`docker-compose.yaml`), isolated from the agent workspace (`environment/Dockerfile` and `environment/docker-compose.yaml` never copy or mount `tests/` into `/workspace`).

**`main`** runs the submission as uid 10001. It sees:
- the pinned FASTA (`reference/GRCh38.p13.genome.fa`);
- `public/test.parquet` and `public/junction_tracks.parquet`;
- `public/perturbations.parquet` (the variant and CRISPRi perturbation queries).

It runs `infer.sh` and then `perturb.sh` within one shared time budget. It then validates both
outputs: example IDs and order, fixed shapes, dtypes, finite values, sparse-list limits, no
symlinks, and the 80-GiB prediction limit. Finally it copies them into a sealed snapshot with
SHA-256 manifests. An invalid package gets `score = reward = 0.0` (`invalid_submission = 1.0`) with
`status: invalid_submission`.

**`grader`** never executes submission code. It waits for the sealed snapshot and checks its
hashes. Test targets, junction scoring queries, splice positions and variant labels are mounted only
here. It then scores:
- **Axis 1 (`tracks_axis`, weight `0.5`)** with `environment/evaluation/scoring.py`: per-track Pearson $r$ in `asinh` space across the 11 output families, grouped into the 5 equal-weight `TRACK_GROUPS` modalities (`Open chromatin`, `Transcription`, `Protein binding and histones`, `Splicing`, `Chromatin contacts`);
- **Axis 2 (`variants_axis`, weight `0.5`)** with `environment/evaluation/variant_scoring.py` and `scoring.py:variants_axis_score`: 12 non-redundant variant/perturbation metrics calibrated against frozen constant-predictor chance anchors (`GENOME_CONSTANT`), **excluding redundant `eqtl_sign`**, averaged equally across 5 biological branches (`eQTL`, `Other molecular QTL`, `CRISPRi enhancer–gene`, `Clinical and functional assays`, `Human traits`).

It writes `report.json` and `reward.json` (`score`, `reward`, `signed_score`, `tracks_axis`, `variants_axis`, `overall_mean_family_pearson`), where `score = reward = clip(0.5 * tracks_axis + 0.5 * variants_axis, 0.0, 1.0)`.

Run the source tests with `bash tests/run_unit_tests.sh`; the baseline's tests are in
`environment/baseline/alphagenome-reference/reference/alphagenome_50m/tests`.

`python tests/measure_prediction_size.py` measures the v3 dense output size. That size motivated
the sparse v4 contract: about 636 MB per example, against about 11.5 MB per example under v4.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary task score (`reward.json["score"]` and `reward.json["reward"]`, bounded in $[0, 1]$) weights **held-out functional genomic track prediction** and **genetic variant / perturbation effect prediction** equally:

$$S_{\text{genome}} = \operatorname{clip}_{[0, 1]}\!\Big(0.5 \cdot S_{\text{tracks}} + 0.5 \cdot S_{\text{variants}}\Big)$$

Both axes directly implement established protocols from the regulatory genomics literature:
- **Axis 1 (`tracks_axis`, weight `0.5`) — 5-modality held-out profile correlation**: Following *Enformer* (Avsec et al., *Nature Methods* 2021, DOI: `10.1038/s41592-021-01252-x`), *Borzoi* (Linder et al., *Nature Genetics* 2025), and *AlphaGenome* (Avsec et al., *Nature* 2025/2026, DOI: `10.1038/s41586-025-10014-0`), held-out interval accuracy is measured by per-track Pearson correlation $r$ over valid positions in variance-stabilized (`asinh`) space on `FOLD_0`'s test fold (1,000 intervals across 5,930 human tracks). To prevent modalities split into multiple readout tensors (e.g., 3 splicing families or 3 transcription assays) from dominating single-tensor modalities (e.g., 3D `contact_maps`), the 11 family means are grouped hierarchically into **5 equal-weight biological modalities**:
  1. `Open chromatin`: mean(`atac`, `dnase`)
  2. `Transcription`: mean(`procap`, `cage`, `rna_seq`)
  3. `Protein binding and histones`: mean(`chip_tf`, `chip_histone`)
  4. `Splicing`: mean(`splice_sites`, `splice_usage`, `junctions`)
  5. `Chromatin contacts`: `contact_maps`
- **Axis 2 (`variants_axis`, weight `0.5`) — 5-branch chance-calibrated variant & perturbation skill**: As demonstrated in *Enformer* (*Nature Methods* 2021), *AlphaGenome* (*Nature* 2025, `10.1038/s41586-025-10014-0`), and *TraitGym* (Benegas et al., 2025), bulk track Pearson correlation saturates far earlier than single-nucleotide causal variant and enhancer perturbation accuracy—the central scientific application of sequence-to-function models. Each non-redundant variant metric $m_k$ is computed by the upstream published scorer (tissue-weighted AUROC/AUPRC/Spearman from *AlphaGenome*'s released evaluation protocol; chromosome-weighted AUPRC over 1:9 matched controls from *TraitGym*) and chance-calibrated against the exact frozen constant-predictor baseline $b_k$ (`GENOME_CONSTANT`): $v_k = (m_k - b_k) / (1 - b_k)$, so a constant sequence-insensitive predictor scores exact `0.0`. The 12 calibrated evaluations are aggregated into **5 equal-weight biological branches**:
  1. `eQTL`: mean(`eqtl_causality`, `eqtl_coefficient`)
  2. `Other molecular QTL`: mean(`sqtl_causality`, `caqtl`, `bqtl_spi1`)
  3. `CRISPRi enhancer–gene`: `enhancer_gene` (ENCODE-rE2G enhancer masking)
  4. `Clinical and functional assays`: `0.5 * mean(clinvar_noncoding, clinvar_splice_region, clinvar_missense) + 0.5 * mfass`
  5. `Human traits`: mean(`traitgym_mendelian`, `traitgym_complex`)

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **Why `eqtl_sign` Is Excluded from the Primary Score (Retained as Diagnostic Only)**: On fine-mapped eQTLs, `eqtl_sign` (binary direction AUROC) and `eqtl_coefficient` (signed tissue-weighted Spearman rank correlation of $\Delta\log$ exon expression against continuous effect size $\beta$) read out the exact same signed allelic prediction vector (`variant_scoring.py:169`). Including both in the `eQTL` branch would double-count eQTL directionality relative to `eqtl_causality`; `eqtl_coefficient` strictly subsumes directional sign while also testing quantitative magnitude ordering.
2. **Why Multi-Species Conservation Tools Are Excluded from Pure-Sequence Reference Baselines**: Variant pathogenicity ensembles and multi-species alignment models (such as multi-species phyloP/phastCons conservation annotation pipelines or 100-vertebrate MSA models) ingest cross-species evolutionary conservation tracks that are prohibited under this benchmark's single-genome human DNA-only contract (`GRCh38.p13` FASTA only). Accordingly, pure-sequence reference comparisons report only single-genome sequence-to-function models: *AlphaGenome*'s released variant predictions (which provide partial pure-sequence branch coverage across 4 of the 5 variant branches—10 of the 12 calibrated evaluations, with no released *TraitGym* predictions) and *Borzoi* / *Enformer* on *TraitGym*.
3. **Why No Ad-Hoc Per-Modality Floors or Parameter Caps Are Imposed**: Equal arithmetic averaging across the 5 track modalities and 5 variant branches already penalizes neglect of any biological readout (such as sparse splicing junctions or distal enhancer perturbations) linearly and transparently without brittle step-function cliffs. Likewise, model architecture and parameter count are bounded naturally by the fixed training and 4-hour single-H100 inference budgets rather than artificial parameter caps.

### 3. Diagnostic Signals Retained in report.json

While `reward.json["score"]` reports the locked two-axis V4 score, `report.json` preserves full granular diagnostics for error analysis:
- **Track-level and family-level diagnostics**: per-family `mean_pearson`, `valid_tracks`, `per_track_pearson`, `values_per_track`, sparse splicing `missing_prediction_ratio` (unlisted hidden splice-usage sites and junction query pairs), 5-modality `track_modalities`, and the legacy ungrouped 11-family mean `overall_mean_family_pearson`.
- **Variant and perturbation diagnostics**: all 13 raw uncalibrated evaluation metrics under `report["variant_effects"]` (**including diagnostic `eqtl_sign`**), row and unique-perturbation counts, pure single-genome sequence-to-function reference anchors (`AlphaGenome` on 4/5 branches; `Borzoi` and `Enformer` on `TraitGym`), plus `report["ssi_v4"]` containing `signed_score`, `tracks_axis`, `variants_axis`, all 5 `branches`, and all 12 `calibrated_variants`.

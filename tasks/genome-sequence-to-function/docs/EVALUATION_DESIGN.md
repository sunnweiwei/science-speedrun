# Evaluation design (v4)

The verifier measures two things on one model:
- how well it predicts held-out functional-genomics tracks, which is the reward;
- what it says about genetic variants, which is where such models are used and where they are
  furthest from solved.

Both follow published protocols. Every reported number has an anchor computed on the same rows.

## 1. Track prediction (reward)

- **Data.** 1,000 test intervals of fold 1 (AlphaGenome `FOLD_0`), with full-precision targets
  and all 5,930 human tracks.
- **Metric.** Pearson per track over valid targets in `asinh` space, averaged within each of
  eleven families, then equally across families (`overall_mean_family_pearson`).
  - Dense families are scored at 128-bp bins of the 196,608-bp target window.
  - `splice_sites` is scored per base.
  - Contact maps are scored per 2,048-bp pair.
- **Splice usage and junctions are sparse.** The model lists at most 4,096 sites and 16,384
  junction pairs per example. Usage is gathered at every hidden splice site and junctions at every
  hidden scoring pair. A hidden site or pair that the model did not list counts as a prediction of
  zero. It stays in the metric, so a model cannot raise its score by listing fewer sites.
- **Relation to AlphaGenome.** AlphaGenome reports per-assay Pearson on held-out intervals of the
  same folds. Here the bin is 128 bp for every dense family and the eleven families are weighted
  equally. The number is therefore comparable across submissions to this task, but not to the
  paper's per-assay tables.

## 2. Variant and perturbation effects (reported, not in the reward)

Each query is one *perturbation*: a 1,048,576-bp window and two sequences, A and B. B differs from
A either by one SNV allele or by an N-masked interval. The submission's `perturb.sh` returns both
predictions over a requested slice. The grader turns them into a variant score with the published
scorer of each evaluation (`environment/evaluation/variant_scoring.py`). The scores are computed
on the grader side, so a submission cannot tailor its scores to the metric.

### Evaluations

The first eleven are AlphaGenome's own variant evaluations, from its released tables
(`storage.googleapis.com/alphagenome/evals`, CC-BY 4.0). The last two are TraitGym (Benegas et al.
2025). "Chance" is the positive fraction for AUPRC, 0.5 for AUROC and 0 for Spearman.

| evaluation | question | readout | metric | rows (perturbations) | chance | AlphaGenome on these rows |
|---|---|---|---|---:|---:|---:|
| eQTL sign | direction of a fine-mapped eQTL's effect on its gene | Δ log mean exon RNA-seq, signed | AUROC, tissue-weighted | 48,020 (452) | 0.50 | 0.813 |
| eQTL causality | causal eQTL vs matched non-causal variant | \|Δ log exon RNA-seq\| | AUROC, tissue-weighted | 44,590 (430) | 0.50 | 0.827 |
| eQTL effect size | fine-mapped eQTL coefficient | Δ log exon RNA-seq, signed | Spearman, tissue-weighted | 23,961 (451) | 0 | 0.499 |
| sQTL causality | causal sQTL vs matched variant | max \|Δ splice-site probability\| within ±2.5 kb | AUPRC, tissue-weighted | 53,680 (487) | 0.57 | 0.860 |
| ClinVar non-coding | pathogenic vs benign, deep intronic and UTR | same splice readout | AUPRC | 499 | 0.20 | 0.928 |
| ClinVar splice region | pathogenic vs benign near splice sites | same | AUPRC | 489 | 0.20 | 0.581 |
| ClinVar missense | pathogenic vs benign missense (splicing component only) | same | AUPRC | 491 | 0.20 | 0.357 |
| MFASS | exon-skipping variants from a massively parallel splicing assay | same | AUPRC | 316 | 0.20 | 0.813 |
| caQTL | causal chromatin-accessibility QTL (African ancestry) | mean \|Δ log2 DNase\| over ±250 bp | AUPRC | 477 | 0.20 | 0.762 |
| SPI1 bQTL | causal SPI1-binding QTL | mean \|Δ log2 SPI1 ChIP\| over ±250 bp | AUPRC | 494 | 0.20 | 0.804 |
| CRISPRi enhancer–gene | does silencing this enhancer change this gene (ENCODE-rE2G) | \|Δ log exon RNA-seq\| when the enhancer is masked to N | AUPRC | 325 | 0.20 | 0.869 |
| TraitGym Mendelian | causal variant for a Mendelian disease vs 9 matched controls | L2 of Δ over all tracks, ±1 kb | AUPRC by chromosome | 784 | 0.10 | Borzoi 0.459, Enformer 0.471, CADD 0.743, GPN-MSA 0.744 |
| TraitGym complex | causal fine-mapped GWAS variant vs 9 matched controls | same | AUPRC by chromosome | 789 | 0.10 | Borzoi 0.290, Enformer 0.335, CADD 0.295, GPN-MSA 0.290 |

### Why these evaluations

- **They are the uses of the model.** AlphaGenome's own evaluations cover expression, splicing,
  chromatin and TF binding, clinical classification and enhancer–gene regulation. They are the
  questions a sequence-to-function model is used to answer.
- **TraitGym adds the hardest, most clinically relevant setting.** It asks for causal variants for
  human disease and traits against matched controls. Sequence-to-function models trail
  conservation-based scores (CADD, GPN-MSA) by a wide margin on the Mendelian set, so a gain there
  would be new.
- **Each evaluation is rigorous on its own terms.** Labels come from fine-mapping, clinical
  curation or perturbation experiments, not from the model's training targets.

### Subsets: rigorous but fast

All 6,587 perturbations take 13,174 forward passes. The tiny reference answered them in 520 s on
one H100, so a large model still fits the verifier's budget.

- **Locus restriction.** AlphaGenome evaluations use only variants whose edited base lies in a
  fold-1 (test) target window, so no evaluated locus had training targets.
  - SPI1 bQTL is the exception: fold 1 has only three positives, so it is drawn genome-wide.
  - TraitGym is also drawn genome-wide: whole match groups (1 positive : 9 controls), 80 groups
    per set; fold 1 alone has too few positives. Its loci may lie in training windows, where the
    model saw targets (not variant labels). This is listed as a limitation in the README.
- **Rare-positive evaluations** use a fixed design of 100 positives and 400 negatives, drawn with
  a fixed seed: ClinVar ×3, MFASS, caQTL, bQTL and enhancer–gene.
- **eQTL and sQTL** are seeded, label-stratified subsets of 500 variants. Every row of each
  selected variant is kept, across all tissues and genes.

### Development split

The agent cannot see the test labels. It gets `variant_evals/valid`, a development split of the
same 13 evaluations with labels and anchors, so it can see what each problem asks and measure
progress on it.
- **Size.** 2,697 perturbations: 40 positives : 160 negatives for the fixed-design evaluations,
  200 variants for eQTL and sQTL, and 30 TraitGym match groups per set. Scoring the tiny model
  takes 255 s on one H100.
- **Loci.** AlphaGenome evaluations are drawn from the validation fold (fold0), the same fold as
  the 200 track-validation intervals. MFASS has fewer than 40 positives there, and bQTL is drawn
  genome-wide as in the test split; both use loci outside the test fold.
- **Disjoint from test.** No variant, enhancer–gene pair or TraitGym match group appears in both
  splits; the builder asserts this.
- **Test split unchanged.** Building the development split left the test split byte-identical.

| evaluation | tiny model (valid) | anchor on the valid rows |
|---|---:|---:|
| eQTL sign / causality / effect size | 0.547 / 0.560 / 0.092 | AlphaGenome 0.826 / 0.833 / 0.413 |
| sQTL | 0.568 | 0.845 |
| ClinVar non-coding / splice region / missense | 0.200 / 0.272 / 0.253 | 0.941 / 0.722 / 0.374 |
| MFASS / caQTL / bQTL / enhancer–gene | 0.188 / 0.320 / 0.265 / 0.480 | 0.734 / 0.808 / 0.817 / 0.959 |
| TraitGym Mendelian / complex | 0.222 / 0.187 | CADD 0.719 / 0.413, Borzoi 0.503 / 0.295 |

### Checks

- `variant_scoring.published_metric` recomputes each AlphaGenome metric from the released
  predictions of the full tables. All eleven match the published numbers to within 0.0004, so the
  metric code is AlphaGenome's protocol.
- The anchors in the table are AlphaGenome's (or TraitGym's) released scores on exactly the
  evaluated rows, scored by the same code.

### Deviations from AlphaGenome's scorers

- **RNA-seq tracks.** GTEx tracks are masked in the public training data, so eQTL and
  enhancer–gene readouts use ENCODE RNA-seq tracks: those of the matching tissue ontology when
  ENCODE has them, otherwise all of them. The AlphaGenome anchors used GTEx tracks.
- **Splicing variants** are scored by the change in splice-site probability only. AlphaGenome's
  scorer also combines usage and junction changes.
- **Resolution.** Readouts are on 128-bp bins (dense) and per base (splice sites).

These deviations apply equally to every submission. They make the anchors a reference point
rather than an identical protocol.

## 3. Time budget

| run | tiny reference (0.23M) | large reference (215M) |
|---|---:|---:|
| `perturb.sh`, 6,587 perturbations, one H100 | 520 s | 582 s |
| full verifier (`infer.sh` + `perturb.sh` + grading) | 43 min | 31 min (`infer.sh` 604 s) |

The verifier's limit is 6 h on one H100.

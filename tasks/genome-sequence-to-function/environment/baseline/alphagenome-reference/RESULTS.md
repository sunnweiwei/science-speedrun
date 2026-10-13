# Results: alphagenome-reference (215.3M parameters)

Reproduced on 2026-10-01 on one 8× H100 80 GB host with the commands in [README.md](README.md):

- `reference_large.yaml`, 14 epochs over all 41,020 training intervals, seed 17;
- 71,792 steps at 0.208 s;
- training took 4 h 9 min on 8 GPUs.

Checkpoint `last.pt`: sha256 `5388434c…77ee`. The package is 2.5 GB.

## Tracks (reward metric)

| family | validation (200, `score.py`) | test (1,000, verifier) |
|---|---:|---:|
| ATAC | 0.487 | 0.488 |
| DNase | 0.540 | 0.564 |
| PRO-cap | 0.473 | 0.484 |
| CAGE | 0.240 | 0.254 |
| RNA-seq | 0.478 | 0.482 |
| TF ChIP | 0.371 | 0.364 |
| histone ChIP | 0.419 | 0.418 |
| splice sites | 0.013 | 0.013 |
| splice usage | 0.028 | 0.023 |
| junctions | 0.000 | 0.000 |
| contact maps | 0.273 | 0.326 |
| **mean over families (reward)** | **0.302** | **0.3106** |

The splicing families are near zero. The splice-site loss is a per-base binary cross-entropy
averaged over 786,432 positions, of which about 0.1 % are sites. Its gradient therefore
vanishes next to the dense losses. Usage and junction outputs depend on the predicted sites.
This is the unchanged reference recipe. Fixing it is part of the research.

## Variant evaluations

The anchor column is AlphaGenome's score, except for TraitGym, which shows CADD, the strongest of
its anchors. Anchors are computed on the same rows.

| evaluation | metric | chance | valid (dev split) | anchor (valid) | test | anchor (test) |
|---|---|---:|---:|---:|---:|---:|
| eQTL sign | AUROC | 0.50 | 0.543 | 0.826 | 0.494 | 0.813 |
| eQTL causality | AUROC | 0.50 | 0.546 | 0.833 | 0.547 | 0.827 |
| eQTL effect size | Spearman | 0 | 0.085 | 0.413 | −0.019 | 0.499 |
| sQTL causality | AUPRC | — | 0.731 | 0.845 | 0.734 | 0.860 |
| ClinVar non-coding | AUPRC | 0.20 | 0.168 | 0.941 | 0.207 | 0.928 |
| ClinVar splice region | AUPRC | 0.20 | 0.201 | 0.722 | 0.209 | 0.581 |
| ClinVar missense | AUPRC | 0.20 | 0.190 | 0.374 | 0.205 | 0.357 |
| MFASS | AUPRC | 0.20 | 0.206 | 0.734 | 0.170 | 0.813 |
| caQTL | AUPRC | 0.20 | 0.147 | 0.808 | 0.178 | 0.762 |
| SPI1 bQTL | AUPRC | 0.20 | 0.186 | 0.817 | 0.186 | 0.804 |
| CRISPRi enhancer–gene | AUPRC | 0.20 | 0.407 | 0.959 | 0.393 | 0.869 |
| TraitGym Mendelian | AUPRC | 0.10 | 0.184 | CADD 0.719 | 0.153 | CADD 0.743 |
| TraitGym complex | AUPRC | 0.10 | 0.187 | CADD 0.413 | 0.192 | CADD 0.295 |

Besides sQTL and enhancer–gene linking, the baseline is near chance on variant effects, although
its track Pearson roughly doubled over the tiny model. Two checks rule out a scoring artifact:
- **Precision of the stored outputs.** Rescoring the development split from float16 outputs
  instead of float32 changes every metric by at most 0.016.
- **Variant placement.** For every SNV query the reference base matches one of the two alleles.
  The predicted A − B difference peaks at the variant's bin, or one 1-kb trunk token away, in 20
  of 24 sampled variants.

## Cost in the verifier (one H100)

| phase | time |
|---|---:|
| `infer.sh`, 1,000 test intervals | 10.1 min |
| `perturb.sh`, 6,587 perturbations | 9.7 min |
| whole verifier run (inference, validation, snapshot, grading) | 31 min |

The development split (2,697 perturbations, `perturb.py --queries` plus `score_variants.py`) took
5.2 min.

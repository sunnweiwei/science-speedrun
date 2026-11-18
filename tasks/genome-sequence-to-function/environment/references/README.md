# References (not verified in this setup)

Background material for model development. Nothing here was re-run in this task's environment;
the verified starting point is `/workspace/baseline/alphagenome-reference`.

| work | what it offers |
|---|---|
| Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome*, Nature 649 (2026) (`10.1038/s41586-025-10014-0`) | the architecture this task follows (U-Net encoder/decoder around a transformer, 1-Mb context, base-pair and 128-bp heads, splice-junction and contact-map heads), the training recipe (pretraining on four folds, then distillation), and the variant scorers covering 4 of the 5 variant branches (10 of the 12 calibrated variant evaluations; no released TraitGym predictions) |
| Linder et al., *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation* (Borzoi), Nature Genetics 57 (2025) | 524-kb context, RNA-seq coverage at 32-bp resolution; the fold split used here and pure single-genome sequence-to-function reference anchors on TraitGym |
| Avsec et al., *Effective gene expression prediction from sequence by integrating long-range interactions* (Enformer), Nature Methods 18 (2021) | 196-kb context transformer; the original variant-effect scoring recipes and pure single-genome sequence-to-function reference anchors on TraitGym |
| Benegas et al., *TraitGym* (2025) | the Mendelian and complex-trait causal-variant benchmarks forming the `Human traits` branch of the reward, reported with pure single-genome sequence-to-function anchors (`Borzoi`, `Enformer`) |

**Pure-sequence scope note.** Multi-species evolutionary conservation and alignment-based variant
scorers (such as multi-species phyloP/phastCons annotation ensembles or 100-vertebrate MSA language
models) depend on cross-species evolutionary inputs prohibited to single-genome DNA-only submissions
in this benchmark; they are therefore excluded from pure-sequence baseline and frontier reference
claims. Likewise, published AlphaGenome released variant tables provide partial pure-sequence
coverage across 4 of the 5 variant branches (excluding `TraitGym`).

Pretrained parameters of these models must not be used if they were trained on this task's
validation or test folds (see the instruction's rules). Architectures, objectives and published
recipes are free to use.

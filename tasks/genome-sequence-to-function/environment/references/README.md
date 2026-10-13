# References (not verified in this setup)

Background material for model development. Nothing here was re-run in this task's environment;
the verified starting point is `/workspace/baseline/alphagenome-reference`.

| work | what it offers |
|---|---|
| Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome*, Nature 649 (2026) | the architecture this task follows (U-Net encoder/decoder around a transformer, 1-Mb context, base-pair and 128-bp heads, splice-junction and contact-map heads), the training recipe (pretraining on four folds, then distillation) and the variant scorers reported next to the reward |
| Linder et al., *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation* (Borzoi), Nature Genetics 57 (2025) | 524-kb context, RNA-seq coverage at 32-bp resolution; the fold split used here |
| Avsec et al., *Effective gene expression prediction from sequence by integrating long-range interactions* (Enformer), Nature Methods 18 (2021) | 196-kb context transformer; the original variant-effect scoring recipes |
| Benegas et al., *TraitGym* (2025) | the Mendelian and complex-trait causal-variant benchmarks reported here, with Borzoi, Enformer, CADD and GPN-MSA scores |

Pretrained parameters of these models must not be used if they were trained on this task's
validation or test folds (see the instruction's rules). Architectures, objectives and published
recipes are free to use.

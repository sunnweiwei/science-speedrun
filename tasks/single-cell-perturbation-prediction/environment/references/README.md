# References

Material we have **not** verified in this setup. Verified packages and their reproduced scores are
in `/workspace/baseline/`.

## Source code in the image

- `arc-state` at `9bbfe78a434a55205e4de834e1ea99f85f7a3add` (installed; `state` CLI): STATE's
  State Embedding (SE) and State Transition (ST) models, including the SE-latent ST variant and
  `FinetuneVCICountsDecoder` that the B2 baseline does not use (its decoder checkpoint paths are
  hard-coded).
- `cell-eval` 0.8.2 and `pdex` 0.3.0 (installed): the official Virtual Cell Challenge scorer.

## No frontier reference point

There is no published number for a fully trained frontier model on this evaluation. B2 uses the
SOTA architecture at a small budget (`state_sm`, 20k steps, a 7,330-gene panel, about 6 GPU-hours).
Arc's released STATE checkpoint was not run zero-shot on these contexts. The VCC 2025 leaderboard
is not directly comparable: `ctx_far` is built from that dataset, but the metric panel and the
normalisation differ.

## Papers

- A. Adduri et al., *Predicting cellular responses to perturbation across diverse contexts with
  State*, bioRxiv 2025. STATE; the held-out-context setting this task follows.
- Arc Institute, Virtual Cell Challenge 2025 and 2026: the metric panel (DES, PDS, MAE) and the
  zero-shot 2026 setting.
- J. M. Replogle et al., *Mapping information-rich genotype-phenotype landscapes with genome-scale
  Perturb-seq*, Cell 2022 (K562).
- A. Nadig et al., Nature Genetics 2025: essential-scale CRISPRi Perturb-seq (the Jurkat training
  context; GEO GSE264667).
- Xaira Therapeutics, X-Atlas/Orion genome-wide Perturb-seq (HCT116, HEK293T), 2025.
- 2026 perturbation-benchmark studies report that deep models regress to cross-context mean-delta
  baselines under context shift; the B1/B2 comparison here reproduces that pattern.

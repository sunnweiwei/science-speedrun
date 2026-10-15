# References (not verified in this setup)

## Uni-Mol2 (Ji et al., arXiv 2406.14969, 2024)

- **Model.** A two-track transformer: an atom track and a pair track with outer-product and triangular updates,
  feature embeddings from RDKit atom/bond features, and 3D pair distances.
- **Pretraining.** Masked atom types, coordinate denoising and pair-distance denoising (weights 1:1:1) on about
  800 M conformations. Sizes 84M, 164M, 310M, 570M and 1.1B.
- **Recipe.** Adam, lr 1e-4, 100k warmup, batch 1,024. The paper's runs are far longer than this task's budget.
- **Findings.** Downstream error falls smoothly with model size, data and compute. QM9 and COMPAS-1D are the
  paper's main fine-tune benchmarks (Tables 4–6).
- **Code and weights.** github.com/deepmodeling/Uni-Mol (`unimol2/`); weights at huggingface.co/dptech/Uni-Mol2.
  The weights are **not** available in this workspace, and the rules exclude external weights.

## Released Uni-Mol2 84M under this task's protocol

The released 84M checkpoint was pretrained on the full corpus for the paper's full schedule. We fine-tuned it with
this task's verifier protocol (15 epochs, seed 0, test MA-RAE):

| group | released 84M | baseline (30k updates) | no pretraining |
|---|---:|---:|---:|
| QM9 | 0.192 | 0.183 | 0.413 |
| COMPAS-1D | 0.121 | 0.166 | 0.662 |
| ExpansionRx | 0.668 | 0.744 | 1.078 |
| ASAP | 0.644 | 0.636 | 1.315 |

With the official 40-epoch recipe on all 107k training molecules, the released model reaches QM9 HOMO/LUMO/gap
agg-MAE 0.0055 Ha. The 5k-molecule, 15-epoch protocol here gives 0.0087 Ha. The ExpansionRx leaderboard winner
reached MA-RAE 0.48 with task-specific models trained on all data.

## Earlier small-scale study (v1 of this task, PR #22, by pj20)

Studied on PCQM4Mv2 HOMO–LUMO gap, with 4.8 M-parameter models and 256k pretraining molecules:

- At matched size and data, the Uni-Mol2 architecture lowered fine-tune error by 7–16 % against Uni-Mol, at every
  label budget.
- Pretraining helped most where labels were scarcest.
- Within the Uni-Mol family, method changes and 50× more molecules did not move transfer; compute did.
- When compute is short, the recipe decides. Uni-Mol2's coordinate head learned little at that scale.

These are hints, not results at this task's scale.

## Other directions reported in the literature

- Denoising pretraining variants: fractional denoising and torsion-aware noise (Frad, SliDe).
- Multi-conformer objectives, and contrastive 2D–3D alignment (GraphMVP, 3D Infomax).
- Larger data mixtures and curriculum over molecule size.
- Longer effective training through throughput: fused kernels, bf16, larger batches with scaled learning rates.

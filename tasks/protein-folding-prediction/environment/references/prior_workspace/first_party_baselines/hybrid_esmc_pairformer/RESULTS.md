# Hybrid ESMC-6B + Pairformer 50M — frozen-100K pass 1

This is the completed first-party hybrid baseline. It combines the frozen,
pinned official ESMC-6B mixed-256 representation with the released Protenix
Pairformer/diffusion graph. It is not a SOTA claim: the randomly initialized
48.76M folding model received one common-data pass with MSA and templates
disabled.

## Training and bundle provenance

| Item | Value |
|---|---:|
| Frozen global train split | 98,977 examples |
| Native all-track projection | 41,219 unique examples |
| DDP exposures | 41,223 (4 deterministic padding repeats) |
| Optimizer steps | 5,889 on 7 H100 GPUs |
| Deployment folding parameters | 48,763,327 |
| Frozen ESMC feature coverage | 91,311 unique sequences; 11.97 GB |
| Training seed | 20260802 |
| Final train loss | 2.03992 |
| Upstream validation complex lDDT, ordinary / EMA | 0.60250 / 0.57767 |
| Runtime digest | `sha256:417586a53e2f03fc1ca066a43dae19e1c007f0bbf708958a9c61a41b7bba82c5` |
| ModelBundle | `/workspace/data/models/hybrid_esmc_pairformer_50m/bundles/foldingtrain100k-pass1-v1` |

The folding parameter count includes Protenix's official `linear_esm`
projection and excludes the separately frozen ESMC-6B. The feature projection
audits every protein entity in the complete native bioassembly because the
released stochastic crop may select any of them. Missing features fail closed;
they are never replaced by zeros or an online model.

The EMA ModelBundle passed inference with networking disabled, the bundle
mounted read-only, and no FoldBench ground truth mounted. The controlled
evaluator then completed all 99 requests with no inference errors. Two issues
found by the isolation gate—an incorrect offline Hugging Face cache root and an
upstream model-name heuristic that attempted to fetch unrelated ESM2 weights—
were corrected at the deployment boundary without changing trained weights.

## Controlled FoldBench

| Track | Primary metric | Score | Diagnostics | Coverage |
|---|---|---:|---|---:|
| Protein monomer | mean lDDT | 0.00 | TM 0.61; GDT-TS 0.48; RMSD 9.87 Å | 10/10 |
| RNA monomer | mean lDDT | 0.00 | TM 0.15; GDT-TS 0.19; RMSD 18.61 Å | 10/10 |
| DNA monomer | mean lDDT | 0.00 | TM 0.12; GDT-TS 0.25; RMSD 15.13 Å | 10/10 |
| Protein–protein | DockQ success | 11.11 | iRMSD 14.13 Å; LRMSD 40.11 Å | 10/10 |
| Antibody–antigen | DockQ success | 3.03 | iRMSD 15.36 Å; LRMSD 53.32 Å | 20/20 |
| Protein–peptide | DockQ success | 22.22 | iRMSD 11.12 Å; LRMSD 17.46 Å | 10/10 |
| Protein–RNA | DockQ success | 9.09 | iRMSD 11.89 Å; LRMSD 24.16 Å | 10/10 |
| Protein–DNA | DockQ success | 8.33 | iRMSD 9.43 Å; LRMSD 20.47 Å | 10/10 |
| Protein–ligand | LRMSD/lDDT-PLI success | 0.00 | lDDT-LP 0.65; lDDT-PLI 0.15 | 10/10 |

FoldBench reports DockQ success as a percentage, so the interface success
scores above are percentage points. These limited-data numbers prove the full
common data → official Protenix training → checkpoint → portable hybrid bundle
→ isolated nine-track evaluation path; they do not establish competitive
folding quality.

Canonical report:
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/hybrid_esmc_pairformer_50m/foldingtrain100k-controlled-suite-v1/foldbench_report.json`.

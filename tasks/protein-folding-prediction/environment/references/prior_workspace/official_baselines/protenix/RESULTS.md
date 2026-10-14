# Protenix 50M — frozen-100K pass 1

This is the completed architecture-faithful, common-data Protenix baseline.
It is not a SOTA claim: the randomly initialized 48.68M folding model received
one deterministic pass through its capability projection of the frozen 100K
DatasetPackage, with MSA and templates disabled.

## Training and bundle provenance

| Item | Value |
|---|---:|
| Frozen global train split | 98,977 examples |
| Native all-track projection | 51,758 unique examples |
| DDP exposures | 51,758 |
| Optimizer steps | 7,394 on 7 H100 GPUs |
| Deployment folding parameters | 48,681,151 |
| Training seed | 20260802 |
| Final train loss | 2.27910 |
| Upstream validation complex lDDT, ordinary / EMA | 0.32366 / 0.32055 |
| Runtime digest | `sha256:92061a6b7ec50af53d5a862cc94cdcfd64ecc4e7c06a118d905f01fe4a9201a6` |
| ModelBundle | `/workspace/data/models/protenix_pairformer_50m/bundles/foldingtrain100k-pass1-v1` |

Training uses the released Protenix tiny graph, official losses, optimizer,
scheduler, permutation logic, EMA, checkpoint loading, and inference runner.
The common frozen DatasetPackage replaces the upstream data mixture. A
physical-GPU collision interrupted the original run after step 3999; the
ordinary checkpoint, matching EMA checkpoint, optimizer/scheduler state,
step counter, and deterministic sampler offset were restored exactly.

The exported EMA ModelBundle passed offline, read-only inference with no
source-tree or ground-truth mount. The controlled evaluator completed all 99
requests with no inference errors.

## Controlled FoldBench

| Track | Primary metric | Score | Diagnostics | Coverage |
|---|---|---:|---|---:|
| Protein monomer | mean lDDT | 0.00 | TM 0.30; GDT-TS 0.16; RMSD 17.85 Å | 10/10 |
| RNA monomer | mean lDDT | 0.00 | TM 0.18; GDT-TS 0.25; RMSD 18.06 Å | 10/10 |
| DNA monomer | mean lDDT | 0.00 | TM 0.14; GDT-TS 0.29; RMSD 16.41 Å | 10/10 |
| Protein–protein | DockQ success | 0.00 | iRMSD 16.67 Å; LRMSD 45.03 Å | 10/10 |
| Antibody–antigen | DockQ success | 0.00 | iRMSD 14.98 Å; LRMSD 51.22 Å | 20/20 |
| Protein–peptide | DockQ success | 0.00 | iRMSD 15.08 Å; LRMSD 20.53 Å | 10/10 |
| Protein–RNA | DockQ success | 0.00 | iRMSD 15.82 Å; LRMSD 29.36 Å | 10/10 |
| Protein–DNA | DockQ success | 0.00 | iRMSD 14.05 Å; LRMSD 30.54 Å | 10/10 |
| Protein–ligand | LRMSD/lDDT-PLI success | 0.00 | lDDT-LP 0.22; lDDT-PLI 0.03 | 10/10 |

These low values are consistent with a small randomly initialized model and a
single limited-data pass. They demonstrate the complete data → official
training → resumable checkpoint → portable bundle → isolated nine-track
evaluator path; they do not demonstrate competitive folding quality.

Canonical report:
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/protenix_pairformer_50m/foldingtrain100k-controlled-suite-v1/foldbench_report.json`.

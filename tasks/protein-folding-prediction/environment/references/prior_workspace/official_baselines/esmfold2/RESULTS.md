# ESMFold2-Fast 50M — frozen-100K all-track pass 1

This is the completed architecture-faithful, common-data result for the
ESMFold2-Fast branch. It is not a SOTA claim: the folding model received one
controlled pass through its capability projection of the frozen 100K
DatasetPackage. The ESMC-6B encoder was frozen and excluded from the 50M
folding-parameter budget.

## Training and bundle provenance

| Item | Value |
|---|---:|
| Frozen global train split | 98,977 examples |
| Native all-track projection | 56,363 unique examples |
| DDP exposures | 56,364 (1 deterministic padding repeat) |
| Optimizer steps | 8,052 on 7 H100 GPUs |
| Deployment folding parameters | 49,198,301 |
| Training seed | 20260802 |
| Final EMA validation loss | 0.2476006 on 28 examples |
| Runtime digest | `sha256:4537af2631e6b266350f59c63b52ddc0b8aea59d759163705d4d9efd21568153` |
| ModelBundle | `/mnt/localssd/${USER}/protein-folding-data/models/esmfold2_fast_50m/bundles/foldingtrain100k-alltrack-pass1-v3` |

The ModelBundle passed offline, read-only inference without a source-tree or
ground-truth mount. It maps the complete public protein/RNA/DNA/ligand target
through Biohub's `StructurePredictionInput`; it does not drop context chains or
restrict evaluation to protein monomers.

## Controlled FoldBench

The official FoldBench metric is listed first. Secondary diagnostics preserve
continuous signal when an official thresholded metric is zero. TM-score,
mean DockQ and lDDT-PLI are higher-is-better; iRMSD is lower-is-better.

| Track | Official metric | Score | Secondary diagnostic | Coverage |
|---|---|---:|---:|---:|
| Protein monomer | mean lDDT | 0.00 | TM-score 0.26 | 10/10 |
| RNA monomer | mean lDDT | 0.00 | TM-score 0.05 | 10/10 |
| DNA monomer | mean lDDT | 0.00 | TM-score 0.04 | 10/10 |
| Protein–protein | DockQ success (%) | 11.11 | mean DockQ 0.099 (9 finite) | 10/10 |
| Antibody–antigen | DockQ success (%) | 22.58 | mean DockQ 0.128 (31 finite) | 20/20 |
| Protein–peptide | DockQ success (%) | 40.00 | mean DockQ 0.2157 (10 finite) | 10/10 |
| Protein–RNA | DockQ success (%) | 27.27 | iRMSD 11.02 Å | 10/10 |
| Protein–DNA | DockQ success (%) | 41.67 | iRMSD 11.22 Å | 10/10 |
| Protein–ligand | RMSD/lDDT-PLI success (%) | 0.00 | lDDT-PLI 0.38 | 10/10 |

Inference produced 99/99 predictions with zero errors and zero unsupported
requests. The external evaluator scored every controlled target at 100%
coverage. Ground truth was absent during inference and mounted only inside the
separate scoring container.

Canonical report:
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/esmfold2_fast_50m/foldingtrain100k-controlled-suite-alltrack-v3/foldbench_report.json`.

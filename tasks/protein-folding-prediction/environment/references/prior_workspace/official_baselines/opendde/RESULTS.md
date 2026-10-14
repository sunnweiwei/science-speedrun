# OpenDDE 50M — frozen-100K pass 1

This is the completed architecture-faithful, common-data OpenDDE baseline.
It is not a SOTA claim: the randomly initialized 48.75M folding model received
one deterministic pass through its capability projection of the frozen 100K
DatasetPackage, with MSA and templates disabled.

## Training and bundle provenance

| Item | Value |
|---|---:|
| Frozen global train split | 98,977 examples |
| Native all-track projection | 55,627 unique examples |
| DDP exposures | 55,629 (2 deterministic padding repeats) |
| Optimizer steps | 7,947 on 7 H100 GPUs |
| Deployment folding parameters | 48,746,962 |
| Training seed | 20260802 |
| Final EMA validation loss | 0.371716 on 28 examples |
| Runtime digest | `sha256:ffe8faa908483a7f6dd6838016a8ffb669e5d20f66d6e4a071090f0f502624f4` |
| ModelBundle | `/workspace/data/models/opendde_structural_50m/bundles/foldingtrain100k-pass1-v1` |

The trainable network is the released OpenDDE graph scaled only through public
configuration dimensions. The public release does not include the original
trainer, so EDM/distogram losses, optimizer, schedule, EMA and validation are
the transparent project-owned `foldingtrain_edm_v1` reconstruction. Released
activation checkpointing, BF16 execution and attention chunking are retained.

The exported EMA ModelBundle passed offline, read-only inference with no
source-tree or ground-truth mount. The controlled evaluator completed all 99
requests with no inference errors.

## Controlled FoldBench

| Track | Primary metric | Score | Diagnostics | Coverage |
|---|---|---:|---|---:|
| Protein monomer | mean lDDT | 0.00 | TM 0.23; GDT-TS 0.10; RMSD 17.11 Å | 10/10 |
| RNA monomer | mean lDDT | 0.00 | TM 0.11; GDT-TS 0.15; RMSD 17.47 Å | 10/10 |
| DNA monomer | mean lDDT | 0.00 | TM 0.09; GDT-TS 0.17; RMSD 11.87 Å | 10/10 |
| Protein–protein | DockQ success | 0.00 | iRMSD 14.12 Å; LRMSD 35.12 Å | 10/10 |
| Antibody–antigen | DockQ success | 15.15 | iRMSD 11.74 Å; LRMSD 48.35 Å | 20/20 |
| Protein–peptide | DockQ success | 10.00 | iRMSD 13.11 Å; LRMSD 17.55 Å | 10/10 |
| Protein–RNA | DockQ success | 0.00 | iRMSD 13.09 Å; LRMSD 21.54 Å | 10/10 |
| Protein–DNA | DockQ success | 25.00 | iRMSD 12.21 Å; LRMSD 27.96 Å | 10/10 |
| Protein–ligand | LRMSD/lDDT-PLI success | 0.00 | lDDT-LP 0.24; lDDT-PLI 0.18 | 10/10 |

FoldBench reports DockQ success as a percentage; therefore 15.15, 10.00 and
25.00 above are percentage points, not DockQ values. These limited-data scores
prove the complete data → training → checkpoint → portable bundle → isolated
nine-track evaluation path; they do not establish competitive model quality.

Canonical report:
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/opendde_structural_50m/foldingtrain100k-controlled-suite-v1/foldbench_report.json`.

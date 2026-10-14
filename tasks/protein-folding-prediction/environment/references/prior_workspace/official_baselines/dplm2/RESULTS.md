# DPLM-2 50M — frozen-100K pass 1

This is the first completed architecture-faithful, common-data result for the
DPLM-2 branch. It is not a SOTA claim: a 47.05M deployment model received one
controlled pass through each of the project-owned 50M MLM, DPLM, and DPLM-2
folding projections. The folding stage retains ByteDance's released DPLM-2
non-data recipe and uses its official structure tokenizer/decoder.

## Training and bundle provenance

| Item | Value |
|---|---:|
| Frozen global train split | 98,977 examples |
| DPLM-2 native folding projection | 42,912 unique protein monomers |
| DDP folding exposures | 47,781 (4,869 deterministic padding repeats) |
| Folding optimizer steps | 369 on 7 H100 GPUs |
| Deployment parameters | 47,053,446 |
| Training seed | 20260802 |
| Runtime | `protein-folding:dplm2-runtime-v6` |
| Runtime digest | `sha256:1ae6f5ef83d322e9f8130fa109d3426cff4593ca717c546f628d3fe40cb829a7` |
| ModelBundle | `/workspace/data/models/dplm2_50m/bundles/foldingtrain100k-pass1-v6` |

The ModelBundle passed offline inference without a source-tree mount. The
FoldBench run used the bundle's merged model and frozen official structure
tokenizer, one sample, seed 20260802, and 500 DPLM-2 diffusion iterations.

## FoldBench

| Track | Native metric | Score | Coverage | Status |
|---|---|---:|---:|---|
| Protein monomer | mean lDDT | 0.0614384384 | 333/334 | supported subset |
| RNA monomer | mean lDDT | — | 0/15 | N/S |
| DNA monomer | mean lDDT | — | 0/14 | N/S |
| Protein–protein | DockQ success | — | 0/279 | N/S |
| Antibody–antigen | DockQ success | — | 0/172 | N/S |
| Protein–peptide | DockQ success | — | 0/51 | N/S |
| Protein–RNA | DockQ success | — | 0/70 | N/S |
| Protein–DNA | DockQ success | — | 0/330 | N/S |
| Protein–ligand | LRMSD/LDDT-PLI success | — | 0/558 | N/S |

`8uxt-assembly1` is the only protein-monomer capability rejection: its 1,596
residues exceed the released DPLM-2 1,022-residue input limit. The supported
subset therefore has 99.70% coverage. The score must not be presented as a
334/334 full-track result.

Official FoldBench/OpenStructure diagnostics on the same 333 targets are mean
GDT-TS 0.0983753754, mean TM-score 0.1772942943, and mean RMSD 32.8470991 Å.
The low score is consistent with the one-pass validation losses; it proves the
full data → official training → portable bundle → external evaluator path, not
that this small training budget is competitive.

Canonical evaluator output:
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/dplm2_50m/foldingtrain100k-pass1-seed20260802-v1`.

The common 99-assembly controlled suite independently re-scored all ten
selected protein monomers with the pinned official evaluator and suite seed
101: lDDT 0.06, GDT-TS 0.10, TM-score 0.19, and RMSD 31.18 Å. Coverage is
10/10 for protein monomer; the other eight tracks are N/S by declared native
capability. Inference was offline, ground-truth-free, and used the same suite
manifest hash as the other four candidates. Its audited report is
`/mnt/localssd/${USER}/protein-folding-eval/foldbench-v1/runs/dplm2_50m/foldingtrain100k-controlled-suite-audited-v6/foldbench_report.json`.

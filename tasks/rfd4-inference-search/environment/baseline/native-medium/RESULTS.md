# Native Medium measurements

## Four-track Figure 2 local baselines (`rfd4-a6000-warm-v1`)

Independent RTX A6000 measurements (`200` fresh attempts per target, common AF3 5-recycle refolding with seed `727000`, Foldseek `10-941cd33` clustering at `TM-score >= 0.5` on `designed_chain_path`) cover **all four official Figure 2 suites** (`medium/front_loaded` for protein and DNA binding; `medium/midpeak` for small-molecule binding and enzyme scaffolding):

| Track | Targets (Attempts) | Local Passes | Local Clusters (%) | Paper Clusters (%) | Diff (pp) | Local Warm Gen (att/GPU-h) | Local Total incl. AF3 (att/GPU-h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protein binding | 24 (4,800) | 839 | **481 (10.02%)** | 466 (9.71%) | +0.31 | 91.3 | 68.0 |
| Small-molecule binding | 77 (15,400) | 4,543 | **2,052 (13.32%)** | 1,937 (12.58%) | +0.75 | 444.4 | 217.3 |
| Enzyme scaffolding | 37 (7,400) | 933 | **766 (10.35%)** | 785 (10.61%) | -0.26 | 274.6 | 157.9 |
| DNA binding | 3 (600) | 13 | **12 (2.00%)** | 18 (3.00%) | -1.00 | 460.5 | 211.6 |

[Four-track Figure 2 Medium receipt](reproduction/figure2-medium.json) · [DNA Medium cohort receipt](reproduction/dna-medium.json) · [Per-target comparison CSV](../../references/overall/measured/logs/per_target_comparison.csv)

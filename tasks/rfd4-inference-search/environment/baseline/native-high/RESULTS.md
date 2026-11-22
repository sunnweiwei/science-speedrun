# Native High measurements

## Four-track Figure 2 local baselines (`rfd4-a6000-warm-v1`)

Independent RTX A6000 measurements (`200` fresh attempts per target, common AF3 5-recycle refolding with seed `727000`, Foldseek `10-941cd33` clustering at `TM-score >= 0.5` on `designed_chain_path`) are complete for `Enzyme scaffolding` (`high/midpeak`, 37/37 targets) and `DNA binding` (`high/flat`, 3/3 targets), with `Protein binding High` (18/24) and `Small-molecule binding High` (58/77) completing in the background:

| Track | Targets (Attempts) | Local Passes | Local Clusters (%) | Paper Clusters (%) | Diff (pp) | Local Warm Gen (att/GPU-h) | Local Total incl. AF3 (att/GPU-h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Enzyme scaffolding | 37 (7,400) | 1,074 | **876 (11.84%)** | 893 (12.07%) | -0.23 | 231.1 | 142.9 |
| DNA binding | 3 (600) | 19 | **17 (2.83%)** | 21 (3.50%) | -0.67 | 283.0 | 164.3 |

[Figure 2 High receipt](reproduction/figure2-high.json) · [Per-target comparison CSV](../../references/overall/measured/logs/per_target_comparison.csv)

## Unconditional monomer check

On RTX A6000, the public paper-size run generated 150 unconditional monomers, 50 per length. AF3 passed 123/150. Foldseek yielded **83/150 distinct passing clusters (55.3%)**. End-to-end yield including final AF3 was 73.6 distinct passing clusters per GPU-hour. Each length used generation seed 727000 + length and AF3 seed 727000.

[100 residues](reproduction/length-100.json) · [150 residues](reproduction/length-150.json) · [250 residues](reproduction/length-250.json) · [Generation log](reproduction/generation.log)

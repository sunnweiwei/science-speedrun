# Native Low measurements

## Four-track Figure 2 local baselines (`rfd4-a6000-warm-v1`)

Independent RTX A6000 measurements (`200` fresh attempts per target, common AF3 5-recycle refolding with seed `727000`, Foldseek `10-941cd33` clustering at `TM-score >= 0.5` on `designed_chain_path`) cover all four official Figure 2 suites:

| Track | Targets (Attempts) | Local Passes | Local Clusters (%) | Paper Clusters (%) | Diff (pp) | Local Warm Gen (att/GPU-h) | Local Total incl. AF3 (att/GPU-h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protein binding | 24 (4,800) | 672 | **426 (8.88%)** | 371 (7.73%) | +1.15 | 136.0 | 89.5 |
| Small-molecule binding | 77 (15,400) | 4,556 | **2,009 (13.05%)** | 1,867 (12.12%) | +0.92 | 551.1 | 241.6 |
| Enzyme scaffolding | 37 (7,400) | 924 | **753 (10.18%)** | 724 (9.78%) | +0.39 | 323.7 | 160.9 |
| DNA binding | 3 (600) | 15 | **15 (2.50%)** | 17 (2.83%) | -0.33 | 673.1 | 247.1 |

[Four-track Figure 2 Low receipt](reproduction/figure2-low.json) · [Per-target comparison CSV](../../references/overall/measured/logs/per_target_comparison.csv)

## Unconditional monomer check

On RTX A6000, the public paper-size run generated 150 unconditional monomers, 50 per length. AF3 passed 119/150. Foldseek yielded **82/150 distinct passing clusters (54.7%)**. End-to-end yield including final AF3 was 112.5 distinct passing clusters per GPU-hour. Each length used generation seed 727000 + length and AF3 seed 727000.

[100 residues](reproduction/length-100.json) · [150 residues](reproduction/length-150.json) · [250 residues](reproduction/length-250.json) · [Generation log](reproduction/generation.log)

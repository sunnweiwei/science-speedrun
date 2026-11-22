# RFD4-Proteina Figure 2 baseline status and multi-track evaluation

**10 of 16 official baseline points now have complete independent RTX A6000 measurements across all four Figure 2 panels (`Low` 4/4, `Medium` 4/4, `Enzyme High`, and `DNA High`, totaling 64,400 completed attempts across 322 cohorts), with the final 25 `High` cohorts (`Protein High` 18/24, `Small-molecule High` 58/77) actively completing in the background.** All 16 official reference settings (`Low`, `Medium`, `High`, and `Search` across `Protein binding`, `Small-molecule binding`, `Enzyme scaffolding`, and `DNA binding`) are retained as the benchmark baseline frontier in [baseline_frontier_reference.csv](../environment/references/overall/measured/baseline_frontier_reference.csv) and [operating_points.csv](../environment/references/overall/operating_points.csv). [Paper](https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1) · [Coverage JSON](../environment/references/overall/measured/coverage.json) · [Local vs paper CSV](../environment/references/overall/measured/comparison_vs_paper.csv) · [Per-target comparison CSV](../environment/references/overall/measured/logs/per_target_comparison.csv)

## Completed independent local baselines (`rfd4-a6000-warm-v1`)

All four Figure 2 suites now have verified end-to-end local baselines (200 fresh attempts per target, warm batch-amortized generation timing, common AF3 evaluation with 5 recycles and seed `727000`, Foldseek `10-941cd33` clustering at `TM-score >= 0.5` on `designed_chain_path`):

| Suite | Setting | Targets | Attempts | Paper clusters (%) | Local clusters (%) | Diff (pp) | Local warm gen (att/GPU-h) | Local total incl. AF3 (att/GPU-h) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protein binding | Low | 24 | 4,800 | 371 (7.73%) | **426 (8.88%)** | +1.15 | 136.0 | 89.5 |
| Protein binding | Medium | 24 | 4,800 | 466 (9.71%) | **481 (10.02%)** | +0.31 | 91.3 | 68.0 |
| Small-molecule binding | Low | 77 | 15,400 | 1,867 (12.12%) | **2,009 (13.05%)** | +0.92 | 551.1 | 241.6 |
| Small-molecule binding | Medium | 77 | 15,400 | 1,937 (12.58%) | **2,052 (13.32%)** | +0.75 | 444.4 | 217.3 |
| Enzyme scaffolding | Low | 37 | 7,400 | 724 (9.78%) | **753 (10.18%)** | +0.39 | 323.7 | 160.9 |
| Enzyme scaffolding | Medium | 37 | 7,400 | 785 (10.61%) | **766 (10.35%)** | -0.26 | 274.6 | 157.9 |
| Enzyme scaffolding | High | 37 | 7,400 | 893 (12.07%) | **876 (11.84%)** | -0.23 | 231.1 | 142.9 |
| DNA binding | Low | 3 | 600 | 17 (2.83%) | **15 (2.50%)** | -0.33 | 673.1 | 247.1 |
| DNA binding | Medium | 3 | 600 | 18 (3.00%) | **12 (2.00%)** | -1.00 | 460.5 | 211.6 |
| DNA binding | High | 3 | 600 | 21 (3.50%) | **17 (2.83%)** | -0.67 | 283.0 | 164.3 |

## Remaining reference settings (completing in background / retained in reference frontier)

| Suite | Setting | Targets | Original attempts | Author clustered pass rate | Author generation attempts/GPU-h | Local status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Protein binding | High | 24 | 4,800 | 10.63% | 985.6 | 18/24 complete on RTX A6000 cluster |
| Protein binding | Search | 24 | 4,800 | 14.92% | 57.4 | Reference frontier (`operating_points.csv`) |
| Small-molecule binding | High | 77 | 15,400 | 14.32% | 5,242.3 | 58/77 complete on RTX A6000 cluster |
| Small-molecule binding | Search | 77 | 15,400 | 20.94% | 110.6 | Reference frontier (`operating_points.csv`) |
| Enzyme scaffolding | Search | 37 | 7,400 | 16.64% | 84.9 | Reference frontier (`operating_points.csv`) |
| DNA binding | Search | 3 | 600 | 4.83% | 111.5 | Reference frontier (`operating_points.csv`) |

## Official paired four-track Harbor grading (`rfd4-paired-program-v1`)

Official Harbor verification ([test.sh](../tests/test.sh) and [task.toml](../task.toml)) executes unified `infer.py` submissions across all four tracks via [evaluate_tracks.py](../environment/evaluation/evaluate_tracks.py) alongside live same-GPU paired baselines (`native-low`, `native-medium`, `native-high`):

1. **Matched timing boundary (`rfd4-paired-program-v1`)**: Candidate and paired baselines for each target execute on the exact same dedicated GPU through the exact same per-request `infer.py` subprocess boundary and per-GPU persistent warm AF3 socket server (`--gpus auto` distributes targets across allocated GPUs without sharing any single GPU); subprocess timings are never stamped as `rfd4-a6000-warm-v1`.
2. **Matched sample counts (`--attempts K`)**: Candidates are compared exclusively against paired baselines run at the exact same `K` and target roster, avoiding `20`-vs-`200` Foldseek deduplication distortion.
3. **Strict Pareto advancement & three-budget compatibility**: Distinguishes exact-tie weak nondominance (`on_total_pareto`) from strict frontier advancement (`strictly_advances_total_pareto`, `total_hypervolume_gain_pct`, `dominates_strongest_reference_total`), supports multi-seed repeat SDs (`--repeats R`), and emits per-arm `complete` flags consumed directly by [budget_scorecard.py](../environment/references/overall/budget_scorecard.py) (`rfd4-budget-curve-v1`, 6/15/60 s/slot).

# References for the research workspace

Your goal is to design **one unified inference-time strategy** (`infer.py`) and strictly advance the quality-cost Pareto frontier across all four Figure 2 tracks (`Protein binding`, `Small-molecule binding`, `Enzyme scaffolding`, `DNA binding`) using `evaluation/evaluate_tracks.py`.

- **Four-track baseline frontier (`Low`, `Medium`, `High`, `Search`) and local RTX A6000 reproduction logs**:
  - [Consolidated 16-point baseline frontier table](overall/measured/baseline_frontier_reference.csv) merges all 16 official `Low / Medium / High / Search` reference points with independent local RTX A6000 measurements (`rfd4-a6000-warm-v1`, 10/16 complete covering all 4 `Low`, all 4 `Medium`, `Enzyme High`, and `DNA High`).
  - [Locally reproduced RTX A6000 points](overall/measured/points.csv) · [Local vs paper comparison](overall/measured/comparison_vs_paper.csv) · [Measured four-panel plots](overall/measured/generation.png) · [Coverage status](overall/measured/coverage.json) · [Timing boundary guide](overall/MEASURED.md)
  - [Per-target local vs paper breakdown (322 completed cohorts)](overall/measured/logs/per_target_comparison.csv) · [Campaign progress logs](overall/measured/logs/campaign_progress.jsonl) · [Cluster sync logs](overall/measured/logs/babel_sync.jsonl) · [GPU preflight receipt](overall/measured/logs/preflight_phase0.json)
  - [Three-budget scorecard standard (`rfd4-budget-curve-v1`, 6/15/60 s/slot)](overall/BUDGET_CURVE.md) · [Scorecard builder](overall/budget_scorecard.py)
  - [Author H200 operating points](overall/operating_points.csv) · [External comparator field](overall/comparison_field.csv) · [Author plots](overall/quality-throughput.png)
- **Monomer ablations and search examples**:
  - [Measured S15 sampler operating points](s15/summary.csv)
  - [Native search config example](upstream/inference_beam_example.yaml) · [License](upstream/LICENSE)
  - [Submission and Pareto evaluation contract](../contract.md) · [RFD4-Proteina paper](https://doi.org/10.64898/2026.11.12.766037)

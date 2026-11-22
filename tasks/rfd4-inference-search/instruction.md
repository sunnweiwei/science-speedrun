# Continuously improve inference for RFD4-Proteina

## Task and goal

Develop **one unified inference-time strategy** (`infer.py`) for the supplied frozen RFD4-Proteina model that strictly advances the **quality-cost Pareto frontier** across **all four Figure 2 biomolecular design tracks**:

1. **Protein binding** (`ppi_backbone`, 24 targets)
2. **Small-molecule binding** (`studio_figure_2c_unfixed`, 77 targets)
3. **Enzyme scaffolding** (`mcsa_fixed`, 37 targets)
4. **DNA binding** (`dna_unfixed`, 3 targets)

Given a runtime request specifying an annotated input structure (`input.cif.zst`), output count, seed and wall-time budget, generate structurally diverse designs that pass the official common AlphaFold 3 (AF3) refolding criteria for that task. Distinct Foldseek structural clusters (`TM-score >= 0.5` on the designed chain) containing at least one AF3 passer count once; duplicate hits in the same cluster earn no extra credit, and failed or missing slots remain in the denominator.

Official Harbor verification executes your single `infer.py` package across all four tracks alongside **live same-GPU paired human baselines** (`native-low`, `native-medium`, `native-high`) under an identical per-request subprocess and shared warm-AF3 timing boundary (`rfd4-paired-program-v1`), matched attempt count per target, and held-out seeds/targets. Simply tying a baseline makes a point weakly nondominated (`on_total_pareto = true`) without advancing the frontier; your goal is **strict Pareto advancement** (`strictly_advances_total_pareto = true`, positive `total_hypervolume_gain_pct`, or `dominates_any_reference_total = true`) across all four tracks with one general inference/search algorithm.

## Workspace

The scoring contract also defines a frozen three-budget report (`rfd4-budget-curve-v1`, at `B/10 = 6 s`, `B/4 = 15 s`, and `B = 60 s` per original attempt) so full-budget quality and low-budget quality are reported cleanly side-by-side. Read `/workspace/references/overall/BUDGET_CURVE.md` for the reporting convention.

- `/workspace/instruction.md` describes the multi-track Pareto-frontier research objective.
- `/workspace/contract.md` specifies the `infer.py` request/response interface (`conditioned_design` and `unconditional_monomer`), matched-boundary timing rules and strict Pareto scoring.
- `/workspace/data/` contains the data guide and read-only benchmark inputs (`assets/benchmarks/` for all 141 official four-track targets and `assets/monomers/`), frozen RFD4 weights, AF3 parameters and Foldseek executable.
- `/workspace/baseline/` contains the unified executable starter (`baseline/program/infer.py`), `native-low`, `native-medium` and `native-high` conditioned/monomer presets with measured RTX A6000 results, and monomer branching policies.
- `/workspace/references/` contains the four-track `Low / Medium / High / Search` baseline reference curves (`overall/measured/baseline_frontier_reference.csv`, `overall/measured/points.csv`, `overall/measured/comparison_vs_paper.csv`), per-target local RTX A6000 reproduction data and campaign logs (`overall/measured/logs/per_target_comparison.csv`, `overall/measured/logs/*.jsonl`), the three-budget scorecard (`overall/BUDGET_CURVE.md`, `overall/budget_scorecard.py`), and monomer ablations.
- `/workspace/evaluation/` contains the official four-track paired Pareto evaluator (`evaluate_tracks.py`, supporting `--gpus auto` across all allocated GPUs), auxiliary monomer evaluator (`run.py`), package builder (`reference.py`) and monomer curve exporter (`cost_curve.py`).
- `/workspace/submission/` holds your current complete, runnable package.
- `/workspace/research/` is writable scratch for candidates, logs, measurements and derived data.

## Rules

Use the supplied assets offline. Keep the RFD4 model weights fixed and generate fresh outputs from each request's input structure and seed. Your submission must implement **one unified inference-time strategy** that handles any conditioned biomolecular request (`Protein binding`, `Small-molecule binding`, `Enzyme scaffolding`, `DNA binding`) as well as unconditional monomers through the same `infer.py` entry point.

You may design adaptive recycling schedules, trajectory branching and pruning, intermediate clean-state or rollout scoring, geometry/chemical filters, predictor-guided search, early stopping and batching. All internal search, discarded branches and intermediate scoring count toward your generation GPU cost. Final common AF3 refolding and Foldseek clustering are owned by the evaluator.

## Submission

Publish a complete package that follows `/workspace/contract.md` early. Its entry point is `infer.py --request FILE --output DIRECTORY`. Bundle every helper module and runtime config your code needs besides the supplied assets and installed dependencies.

From `/workspace`, build and publish a starting baseline package:

```bash
candidate=$(mktemp -d /workspace/research/candidate.XXXXXX)
python evaluation/reference.py native-low "$candidate"
bash baseline/publish.sh "$candidate"
```

Evaluate your unified strategy across all four tracks on the balanced 12-target development panel (`3` targets per track) against paired same-GPU baselines (`native-low`, `native-medium`, `native-high`) at the exact same attempt count (`--gpus auto` parallelizes targets across all allocated GPUs while keeping each target's candidate and paired baselines strictly isolated on a single dedicated GPU):

```bash
run=$(mktemp -d /workspace/research/dev4track.XXXXXX)
python evaluation/evaluate_tracks.py --submission "$candidate" --method-name candidate-v1 \
  --panel dev --attempts 20 --gpus auto --references native-low native-medium native-high \
  --work "$run/work" --output "$run/report"
```

Because Foldseek deduplicated pass rate (`clusters / attempts`) changes nonlinearly with sample size, never compare a `20`-attempt `dev` run directly against `200`-attempt archived tables; always compare against the paired baselines run at the same `--attempts` count and timing boundary, and use `--repeats 3` to verify that gains exceed seed-to-seed dispersion (`*_sd`) before publishing with `bash baseline/publish.sh "$candidate"`.

## Continuous research

Start by inspecting `/workspace/references/overall/measured/baseline_frontier_reference.csv`, `/workspace/references/overall/measured/comparison_vs_paper.csv`, and `/workspace/references/overall/measured/logs/per_target_comparison.csv` (322 completed local RTX A6000 target cohorts covering 10 of 16 Figure 2 settings: `Low` 4/4, `Medium` 4/4, `Enzyme High`, and `DNA High`). Analyze where `Low`, `Medium`, `High`, and `Search` gain or waste compute across targets (for instance, naive extrinsic search improves clustered pass rate over `High` but spends 17-47x more generation GPU time). Design inference-time strategies — such as cheap early clean-prediction pruning, adaptive per-step recycling, uncertainty-gated branching, or selective rollout evaluation — that strictly advance the quality-cost frontier across all four tracks, including under tight budgets (`6 s`, `15 s`, `60 s` per slot).

There is no end point. Keep a working submission published, log every hypothesis and its paired four-track metrics (`strictly_advances_total_pareto`, `total_hypervolume_gain_pct`, `dominates_strongest_reference_total`, `pareto_including_af3.png`), validate promising candidates with multi-seed `--repeats`, and iterate continuously to push out the four-track Pareto frontier.

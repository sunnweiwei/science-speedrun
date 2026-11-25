# Continuously improve inference-time search for RFD4-Proteina

## Task and goal

Develop **one unified inference-time search strategy** (`infer.py`) for the supplied frozen RFD4-Proteina model that **maximizes Foldseek-deduplicated pass rate** across **all four Figure 2 biomolecular design tracks** within a practical wall-time search budget:

1. **Protein binding** (`ppi_backbone`, 24 targets; Paper `High` 10.63%, Paper `Search` 14.92%)
2. **Small-molecule binding** (`studio_figure_2c_unfixed`, 77 targets; Paper `High` 14.32%, Paper `Search` 20.94%)
3. **Enzyme scaffolding** (`mcsa_fixed`, 37 targets; Paper `High` 12.07%, Paper `Search` 16.64%)
4. **DNA binding** (`dna_unfixed`, 3 targets; Paper `High` 3.50%, Paper `Search` 4.83%)

Given a runtime request specifying an annotated input structure (`input.cif.zst`), a **fixed output candidate count** (`num_outputs = K`, matching the paper's `K = 200` on official verification and full Figure 2 panels), a random seed, and a per-request wall-time budget (`budget.wall_seconds = K * seconds_per_slot`), generate structurally diverse designs that pass the official common AlphaFold 3 (AF3) refolding criteria for that task. Distinct Foldseek structural clusters (`TM-score >= 0.5` on the designed chain) containing at least one AF3 passer count once; duplicate hits in the same cluster earn no extra credit, and failed or missing slots remain in the `K` denominator.

**Primary objective (`reward`)**: maximize the **four-track equal-weight mean Foldseek-deduplicated pass rate** (`reward = macro_clustered_pct / 100.0`) under the fixed output budget `K = 200` and wall-time search budget **`25 s / output` calibrated on 8 x H100 GPUs** (fitting the full 16-target x 200-output = 3,200-output official verification cleanly inside 4 hours in two full 8-GPU waves). Aim to **significantly surpass all published baselines**, including RFD4-Proteina's own `Search` operating point (`14.33%` macro across all 141 targets; `14.25%` on the 16-target `verify` roster, where Paper `Search` spent 17-47x more compute than `High`). Do **not** hold back quality merely to match `native-low` runtime: use the allowed wall-time search budget effectively (leaving a safe margin against timeouts) while keeping throughput and cost metrics recorded as secondary diagnostics.

## Workspace

Official Harbor verification (`tests/test.sh`) runs candidate-only evaluation on the held-out **16-target `verify` panel** (`4` PPI, `5` small-molecule, `4` enzyme, `3` DNA) at **`K = 200` outputs/target** and **`25 s / output`** (`budget.wall_seconds = 5000.0` per target) so 100% of the 4-hour 8 x H100 budget is dedicated to inference-time search. Optional same-GPU paired baselines (`--references native-low native-medium native-high`) and the frozen three-budget scorecard (`rfd4-budget-curve-v1`, `/workspace/references/overall/BUDGET_CURVE.md`) remain supported for ablations and cost-curve analysis.

- `/workspace/instruction.md` describes the four-track inference-time search objective.
- `/workspace/contract.md` specifies the `infer.py` request/response interface (`conditioned_design` and `unconditional_monomer`), candidate-budget definition, wall-time enforcement, and scoring rules.
- `/workspace/data/` contains the data guide and read-only benchmark inputs (`assets/benchmarks/` for all 141 official four-track targets and `assets/monomers/`), frozen RFD4 weights, AF3 parameters and Foldseek executable.
- `/workspace/baseline/` contains the unified executable starter (`baseline/program/infer.py`), `native-low`, `native-medium` and `native-high` conditioned/monomer presets with measured RTX A6000 results, and monomer branching policies.
- `/workspace/references/` contains the four-track `Low / Medium / High / Search` baseline reference curves (`overall/measured/baseline_frontier_reference.csv`, `overall/measured/points.csv`, `overall/measured/comparison_vs_paper.csv`), per-target local RTX A6000 reproduction data and campaign logs (`overall/measured/logs/per_target_comparison.csv`, `overall/measured/logs/*.jsonl`), the three-budget scorecard (`overall/BUDGET_CURVE.md`, `overall/budget_scorecard.py`), and monomer ablations.
- `/workspace/evaluation/` contains the official four-track evaluator (`evaluate_tracks.py`, supporting `--gpus auto` across all allocated GPUs, candidate-only `--references none`, or paired baselines), auxiliary monomer evaluator (`run.py`), package builder (`reference.py`) and monomer curve exporter (`cost_curve.py`).
- `/workspace/submission/` holds your current complete, runnable package.
- `/workspace/research/` is writable scratch for candidates, logs, measurements and derived data.

## Rules

Use the supplied assets offline. Keep the RFD4 model weights fixed and generate fresh outputs from each request's input structure and seed. Your submission must implement **one unified inference-time search strategy** that handles any conditioned biomolecular request (`Protein binding`, `Small-molecule binding`, `Enzyme scaffolding`, `DNA binding`) as well as unconditional monomers through the same `infer.py` entry point without privileged track labels.

The **output candidate budget** is strictly fixed to `num_outputs` (`K`), which is the exact number of designs forwarded to common AF3 refolding and Foldseek clustering. **Internal search trajectory count is unconstrained**: you may freely combine fast kernel/engine acceleration, adaptive or cross-step recycling schedules, oversampling pools (`M > K`), trajectory branching and progressive pruning, intermediate clean-state or self-consistency scoring, geometry/physics filters, predictor-guided or evolutionary search, and Foldseek/diversity-aware selection. All internal search, discarded candidates, and intermediate scoring run inside your single-GPU process and are strictly bounded by `budget.wall_seconds` (`K * seconds_per_slot`); a timeout or nonzero exit scores zero for that request.

## Submission

Publish a complete package that follows `/workspace/contract.md` early. Its entry point is `infer.py --request FILE --output DIRECTORY`. Bundle every helper module and runtime config your code needs besides the supplied assets and installed dependencies.

From `/workspace`, build and publish a starting baseline package:

```bash
candidate=$(mktemp -d /workspace/research/candidate.XXXXXX)
python evaluation/reference.py native-high "$candidate"
bash baseline/publish.sh "$candidate"
```

Screen candidate strategies rapidly on the 12-target development panel (`--panel dev`, `3` targets/track) at `K = 20` (with optional `--references native-high`), and validate promising candidates at the **paper-matched `K = 200` outputs/target** before publishing (because Foldseek deduplicated pass rate `clusters / K` decreases nonlinearly as `K` grows from `20` to `200`):

```bash
# Fast screening (K = 20)
run=$(mktemp -d /workspace/research/dev20.XXXXXX)
python evaluation/evaluate_tracks.py --submission "$candidate" --method-name search-v1 \
  --panel dev --attempts 20 --seconds-per-slot 25 --gpus auto --references native-high \
  --work "$run/work" --output "$run/report"

# Paper-matched validation (K = 200, directly comparable in quality to Paper High / Search)
run200=$(mktemp -d /workspace/research/dev200.XXXXXX)
python evaluation/evaluate_tracks.py --submission "$candidate" --method-name search-v1 \
  --panel dev --attempts 200 --seconds-per-slot 25 --gpus auto --references none \
  --work "$run200/work" --output "$run200/report"
```

Read `budget.wall_seconds` dynamically on every invocation so your search automatically scales its internal search depth/pool to fit safely within whatever GPU speed and wall-time budget the controller grants (e.g., `25 s/slot` on H100, or scaled budgets on slower development GPUs), with zero timeouts. Publish validated improvements atomically with `bash baseline/publish.sh "$candidate"`.

## Continuous research

Start by inspecting `/workspace/references/overall/operating_points.csv`, `/workspace/references/overall/measured/baseline_frontier_reference.csv`, `/workspace/references/overall/measured/comparison_vs_paper.csv`, and `/workspace/references/overall/measured/logs/per_target_comparison.csv`. Notice that RFD4-Proteina's `Search` operating point lifted 200-attempt macro clustered pass rate from `10.13%` (`High`) to `14.33%` (`Search`), yet spent 17-47x more generation GPU time than `High`. By combining inference acceleration (so each internal trajectory costs a fraction of `High`) with high-recall trajectory search, strong self-consistency / interface predictors, and diversity-preserving submodular selection at `K = 200`, a unified strategy can spend its full `25 s/slot` H100 budget to push four-track clustered pass rate far beyond all previously reported results.

There is no end point. Keep a working submission published, log every hypothesis and its four-track clustered pass rate (`reward`, per-track `clustered_pct`, `archive_200_search_quality_delta_pp` at `K = 200`, and measured GPU seconds), verify gains at paper-matched `K = 200` (where cluster saturation matters most), and iterate continuously to push the four-track clustered pass rate as high as possible within budget.

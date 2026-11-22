# Submission contract

Submit a single unified Python inference program at `/workspace/submission/infer.py`. Include any helper modules and configuration files in the same directory. The controller invokes one offline process per target request on one GPU. The program must use the supplied frozen RFD4-Proteina weights to generate fresh designs from the current request's input CIF and seed. It controls sampling, recycling, search, branching, early stopping and batching. The controller measures wall/GPU cost and owns final common AlphaFold 3 (AF3) refolding, Foldseek structural clustering and paired Pareto frontier scoring across all four Figure 2 tracks (`Protein binding`, `Small-molecule binding`, `Enzyme scaffolding`, `DNA binding`).

## Request and response

The versioned three-budget reporting contract uses `B/10`, `B/4` and `B` (standard caps of `6`, `15` and `60` generation/search seconds per original attempt). A single-run reward describes its own allowance. The frozen three-tier report (`references/overall/budget_scorecard.py`) records full-budget quality, low-budget quality, paired gains and actual measured cost directly from `evaluate_tracks.py` outputs. [Budget curve standard](references/overall/BUDGET_CURVE.md)

Build and test a complete candidate directory under `/workspace/research/`. From `/workspace`, publish it atomically with `bash baseline/publish.sh "$candidate"`.

The entry point is `python infer.py --request FILE --output DIRECTORY`. The output directory exists and starts empty. Resolve input files relative to the request file. `RFD4_MODEL` points to the frozen RFD4 checkpoint directory and `DATA_PATH` points to the read-only assets root. Each request runs in a fresh process.

### Conditioned four-track design requests (`kind = "conditioned_design"`)

Official four-track evaluation (`evaluation/evaluate_tracks.py`, invoked by Harbor's `tests/test.sh`) dispatches a uniform request envelope across all four suites (`ppi_backbone`, `studio_figure_2c_unfixed`, `mcsa_fixed`, `dna_unfixed`) without revealing privileged suite labels:

```json
{
  "version": 1,
  "id": "ppi_backbone/rfd3_pdl1",
  "kind": "conditioned_design",
  "input": {"file": "rfd3_pdl1.cif.zst"},
  "num_outputs": 20,
  "seed": 727201,
  "budget": {"wall_seconds": 1200.0}
}
```

The annotated input mmCIF (`input.file`) carries the standard RFD4/AtomWorks molecular conditioning (target protein backbone + hotspots, small-molecule ligand, unindexed catalytic motif + fixed ligand, or DNA duplex). Because the same `infer.py` package is invoked across every track, the submitted strategy must generalize across all four tracks automatically.

### Unconditional monomer requests (`kind = "unconditional_monomer"`)

```json
{
  "version": 1,
  "id": "example-117",
  "kind": "unconditional_monomer",
  "input": {"length": 117, "file": "monomer-117.cif.zst"},
  "num_outputs": 5,
  "seed": 7341,
  "budget": {"wall_seconds": 180}
}
```

Read `input`, `num_outputs`, `seed` and `budget` on every invocation. Treat `id` as opaque. Exit `0` on completion, or `64` for an unsupported `version` or `kind`.

For slot `0` write `slot-000000.cif`, for slot `1` write `slot-000001.cif`, and so on up to `num_outputs - 1`. Write regular, uncompressed mmCIF files (`<= 8 MiB` per file). For conditioned tasks, preserve input target/motif conditioning annotations as emitted by RFD4 so downstream refolding and motif/interface RMSD checks succeed. Missing, corrupt, or conditioning-violating outputs remain in the `num_outputs` denominator. Timeout or nonzero exit yields zero outputs for that request while still charging elapsed generation time.

## Timing boundaries, sample-count matching, and strict Pareto evaluation

### 1. Matched timing boundaries (`rfd4-paired-program-v1` vs `rfd4-a6000-warm-v1`)

Cost comparisons are valid **only** when candidate and baseline share the exact same hardware and timing boundary. The benchmark keeps two timing profiles strictly separate:

- **`rfd4-paired-program-v1` (live paired `infer.py` evaluation in `evaluation/evaluate_tracks.py` and `tests/test.sh`)**: Both the candidate package and the packaged human baselines (`native-low`, `native-medium`, `native-high` via `--references`) are invoked through the **exact same per-request `infer.py` subprocess boundary** on the **exact same dedicated GPU** and scored through a **per-GPU persistent AF3 batch socket server** (whose per-bucket XLA compilation warmup is excluded from `af3_gpu_s`). When multiple homogeneous GPUs are allocated (`--gpus auto`), targets are distributed across dedicated single-GPU workers so each target's candidate and paired baselines still run sequentially on the exact same GPU with zero GPU sharing.
- **`rfd4-a6000-warm-v1` (archived warm 200-attempt native campaigns in `references/overall/reproduced_targets.csv` and `measured_report.py`)**: Warm, batch-amortized native GPU kernel timings across all 141 targets at 200 attempts/target. `evaluate_tracks.py` never stamps per-request subprocess runs as `rfd4-a6000-warm-v1` and never mixes `rfd4-a6000-warm-v1` or H200 archive timings onto `rfd4-paired-program-v1` Pareto axes.

### 2. Sample-count matching (`--attempts K`) and panels

Foldseek clustered pass rate (`100 * clusters / attempts`) is a nonlinear, sample-size-dependent statistic because structural collisions accumulate as `attempts` grows. Therefore, `20`-attempt screening runs (`--panel dev`) are **never** compared directly against `200`-attempt archived tables; `evaluate_tracks.py` enforces `b.attempts == candidate.attempts` and `b.targets == candidate.targets` across every Pareto comparison:

- **Development panel (`--panel dev`)**: 12 public targets (3 per track in `requests_io.DEV_4TRACK_TARGETS`), default `20` attempts/target, paired against `--references` at the same `20` attempts/target for rapid screening.
- **Official verifier panel (`--panel verify`, default in `tests/test.sh`)**: quartile-balanced held-out target roster (`VERIFY_4TRACK_TARGETS` in `/tests/rfd4_private.py`, target-disjoint from `dev` on PPI, small-molecule, and enzyme suites) with private HMAC seeds (`/private/seed`), paired on the same GPU against `RFD4_REFERENCES` (default `native-low native-high`). Operators can also set `RFD4_VERIFY_PANEL=full RFD4_VERIFY_ATTEMPTS=200` or supply `/private/four_track_suite.json`.
- **Full Figure 2 panel (`--panel full`)**: all 141 official targets (`24` PPI, `77` small-molecule, `37` enzyme/MCSA, `3` DNA), default `200` attempts/target.

### 3. Strict Pareto frontier advancement vs weak nondominance

For each track with completed paired baselines `B` on speed axis `x` (`generation_attempts_h` or `total_attempts_h`) and quality `y = clustered_pct`:

- **`on_generation_pareto` / `on_total_pareto` (weak nondominance)**: no baseline `b in B` has `x(b) >= x(c)` and `y(b) >= y(c)` with at least one strict inequality. **An exact copy of a baseline satisfies weak nondominance (`True`) and must not be interpreted as advancing the frontier.**
- **`strictly_advances_generation_pareto` / `strictly_advances_total_pareto` (strict frontier advancement)**: candidate completed cleanly, meets at least the lowest paired baseline quality floor (`y(c) >= min_{b in B} y(b)`), and strictly beats **every** paired baseline on either speed or quality (`forall b in B: x(c) > x(b) or y(c) > y(b)`). Exact clones and trivial fast zero-quality emitters receive `False` and `0.0` `*_hypervolume_gain_pct`.
- **`dominates_any_reference_*` and `dominates_strongest_reference_*`**: whether the candidate achieves `>=` speed and `>=` clustered pass rate (with at least one strict inequality) relative to any paired baseline or the highest-quality paired baseline (`High`).
- **Multi-seed repeats (`--repeats R`)**: runs `R` independent seed streams for both candidate and paired baselines, computing each repeat's Pareto curve independently before aggregating `mean` and sample standard deviation (`*_sd`).

## Official Harbor outputs and auxiliary monomer tools

Harbor's official verifier (`tests/test.sh`) runs `evaluation/evaluate_tracks.py` and writes flat numeric metrics to `/logs/verifier/reward.json` (`reward` = 4-track mean clustered pass rate in `[0, 1]`, plus `tracks_strictly_advancing_total_pareto`, `macro_total_hypervolume_gain_pct`, and paired cost/quality ratios against the strongest and fastest references), full structured diagnostics to `/logs/verifier/details.json` and `/logs/verifier/report.json`, paired plots/tables to `/logs/verifier/four_track/`, and any sealed `rfd4-a6000-warm-v1` campaign plots to `/logs/verifier/figure2/`. Auxiliary paired monomer verification remains available via `evaluation/run.py` (or `RFD4_VERIFIER_MODE=monomer`).

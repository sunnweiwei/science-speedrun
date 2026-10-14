# Evaluation within a disk allowance

The default verifier evaluates all eight dates, 56 members and 30 leads using
lossless continuation. It scores a consecutive window, retains only continuation
state, the preceding precipitation interval and the three cyclone fields, then
releases that window's forecast arrays. The next invocation advances from the
saved state; it does not regenerate earlier forecasts. All metric endpoints
remain enabled. Each model invocation exits its offline namespace before any
scoring starts; the resumed model sees only its own state and original context.

## Existing trained models: no new training

The stock starter package can be upgraded without touching its weights:

```bash
python wn2/upgrade_inference.py --source /existing/trained-package \
  --output /new/trained-package-v3
```

The tool verifies the original inventory and stock model implementation, copies
weights/statistics, updates only the inference entry point and continuation
helper, and records the original package and unchanged weights hashes. Keep the
original package. A custom model must implement the v3 continuation contract;
it is not silently replaced with the stock starter. New `package_model.py`
packages already support v3.

## Allowance and execution

```bash
python wn2/verify_track.py --package /new/trained-package-v3 \
  --benchmark /prepared/evaluation-resources --output /new/evaluation \
  --disk-allowance-gb 1000 --disk-reserve-gb 25 --budget-seconds 43200
```

GB is decimal (1,000,000,000 bytes). `tests/test.sh` exposes these as
`WEATHER_EVALUATION_DISK_ALLOWANCE_GB` (default 1000),
`WEATHER_EVALUATION_DISK_RESERVE_GB` (default 25),
`WEATHER_EVALUATION_BUDGET_SECONDS` (default 43200),
`WEATHER_EVALUATION_CONCURRENT_CASES` (default 0: as many as fit) and
`WEATHER_EVALUATION_SCORE_WORKERS` (default 0: one per available CPU). Pass them into the verifier
container. The task's default evaluation storage allocation is approximately
1 TB; changing the allowance also requires changing the host/runner allocation.

Preview the plan without a GPU or model execution:

```bash
python wn2/disk_budget.py --package /new/trained-package-v3 \
  --benchmark /prepared/evaluation-resources --disk-allowance-gb 250
```

Planning counts all prepared evaluation inputs, the original and relocated model,
two generations of continuation state, cyclone fields, precipitation boundary
state, and an explicit reserve for images, caches, logs and temporary products.
Window capacity is based on uncompressed float32 forecasts, not a compression
ratio. An insufficient allowance fails before inference. `storage-plan.json`
records the selected windows and bounds. Approximately 250 GB is a useful lower
starting point for the stock model; exact capacity depends on package/input sizes
and reserve. Containers, downloads and any other files outside the measured roots
must fit inside the reserve or be provisioned separately. Materialize only the
evaluation scope when evaluating an existing trained package.

## Concurrency

Cases run side by side, one per GPU, each in its own windows. The planner gives
every concurrent case its own model copies, two state generations and tracker
buffer, shares the benchmark resources and reserve, and splits the rest into
windows. By default it picks the most cases (up to the GPU count) that still leave
at least three leads per window, and otherwise falls back to one case at a time:
at 1 TB, four cases with five-lead windows; at 500 GB, two cases with four-lead
windows. `--concurrent-cases` fixes the number.

Within a window, each of the three window scorers (`scores`, `pooling-scores`,
`spectra`) runs as one process per lead (`--score-chunk-leads`), all drawing from
one pool of `--score-workers` CPU slots shared by every case; the 56 cyclone
trackers of a case use the same pool. Each lead process still loads the whole
window, so 24-hour precipitation at a lead boundary is computed from the same
arrays, and `merge_lead_chunks` rebuilds exactly the report an unsplit window
writes. Splitting changes no computed value beyond floating-point reduction
order: on a real window, lead 24 scored alone matched the unsplit window to a
relative 6e-15 on every CRPS/MSE/variance field, with REV, SEEPS and the truth
digest identical.

The first failure stops every case and is the one `status.json` records.

Measured on four H100 80 GB and 32 CPU cores (2026-09-30): the full eight-case
evaluation took 5.1 h (a window: about 14 min inference, 9 min scoring; the
cyclone step, 8-11 min per case), against about 40 h projected for one case at a
time, where one window's `scores` alone took over an hour on a single core.
The default budget is 12 h. Smaller allowances mean fewer concurrent cases or
shorter windows and more wall time; set the runner's timeout and evaluation
budget consistently. A smaller disk allowance does not grant extra training time.

The verifier measures its output tree during subprocess execution and between
stages, adds fixed inputs/package/reserve, and terminates on excess. It records
the observed peak in `status.json`. This polling monitor is not a filesystem
quota: a misbehaving submission can overshoot between samples. Use a filesystem
quota if an absolute physical cap is required. Checkpoint generations have an
explicit size bound; v3 requires exactly the declared float32 forecast fields.

## Results, failures and debugging

Native, pooled, REV, SEEPS and spectral rows are concatenated in canonical lead
order, not averaged per window. The 24-hour precipitation boundary retains the
previous 12-hour interval. Cyclone detection/stitching still receives the complete
10-day history and 15-day future; only `msl`, thickness and wind are retained
between windows. Source manifests and window report hashes remain auditable.

Successful windows release forecast arrays; failed windows retain their artifacts
for diagnosis. This is not automatic crash recovery: a new evaluation uses a new
output directory. Normal window continuation never recomputes a forecast step.
Raw forecasts cannot be rescored after cleanup.

For a legacy v2 package or full retained outputs, opt out explicitly with
`--legacy-case-mode`. Use `--keep-predictions` to retain arrays; a forecast-only
phase requires both flags. These modes bypass the allowance planner/monitor and
require the documented full-case/full-run provision (up to 8 TiB). They are not
bounded-allowance workflows.

## Validation without training

`check_continuation.py` compares a supplied trained v2 package with its upgraded
v3 copy in fresh inference namespaces. It checks every decoded field bitwise
across a restart (default: two members, hours 12 and 24). Supply `--original`,
`--upgraded`, `--context` and a new `--output`; allocate one suitable GPU and set
the same model runtime as the verifier. This comparison deliberately runs the
two implementations; production continuation does not repeat forecast steps.

`check_windowed_scoring.py` uses existing full forecasts and actual native-grid
scorers to compare window results, including the 24-hour precipitation boundary.
`check_windowed_tracker.py` verifies three windows covering all 30 leads against
historical tracker inputs and reruns the pinned detector/stitcher. The latter
checks both candidate and track file hashes. These tools never train or modify
the original forecasts. Historical slices use hard links and need the validation
output on the same filesystem. Scope and results are written explicitly; these
checks must not be described as a new full eight-date evaluation.

# Contract `timeseries-forecast/v1`

The package is a directory (the agent keeps it at `/workspace/submission`; it may be read at any
moment, so replace it as a whole: assemble the new package elsewhere and swap it in with one rename,
as `baseline/starter/publish.py` does) that the verifier copies to a sealed location and runs, several times, as

    bash infer.sh INPUT_DIR OUTPUT_DIR

inside a kernel namespace (Bubblewrap, `tests/isolation.py`) that contains the
declared runtime, the package (read-only, at `/submission`), one input
directory (read-only) and one output directory: no network, no data
directory, no labels, no workspace, no host `/tmp`, one GPU
(`docs/validation/isolation_probe.json` shows what a canary package could see).
The calls share one wall-clock budget (4 hours for the full test split,
`TIMESERIES_INFERENCE_TIMEOUT_SECONDS` in `tests/test.sh`). Where Docker refuses
the network namespace the sandbox is built without it and the container's own
no-network policy keeps the package offline; where no namespace works the
verifier runs the package as an unprivileged uid while the labels, the data
(`/private`) and `/tests` stay root-only (`isolation: ["uid"]`); only with an
explicit operator opt-out does it run a plain child process and mark the result
untrusted. `reward.json` names the mode. The package must contain:

- `infer.sh` (bash entry point; the working directory is the package root);
- `model_manifest.json`: `{"name": ..., "contract": "timeseries-forecast/v1",
  "parameters": <int>, "pretrained_assets": [], "sources_used": [...],
  "notes": ...}`. `sources_used` lists the pretraining directories used
  (`"gifteval_pretrain/<dir>"`) and any self-generated synthetic data as
  `"synthetic:<description>"`;
- model code, configuration, weights, normalization statistics and every other
  runtime asset. Inference must be deterministic (the verifier reruns one
  cohort and compares bit for bit).

The runtime is the verifier image's: Python 3.12 with the packages pinned in
`runtime/requirements.txt` (PyTorch 2.14 with CUDA 12.6, NumPy, h5py, pandas,
SciPy, pyarrow, huggingface-hub, einops, safetensors, pytest; at
`/opt/runtime/requirements.txt` in the environment). Nothing else is installed.

## Input

**Why several calls.** GIFT-Eval's test windows roll: window k's labels are the
tail of window k+1's context, and a dataset's short-term contexts run into its
medium- and long-term test regions. Staged into a single call, 95.6% of the
held-out label points would sit inside another instance's context on the same
timeline (92.5% inside another window of the same series). So every instance
carries a `cohort` id and one call receives one cohort. Call c holds, for each
timeline, the instances at its c-th cutoff timestamp. A timeline is a dataset
directory, with every item, frequency and term staged from it
(`electricity/15T`, `/H`, `/D`, `/W` are aggregates of the same meters;
`kdd_cup_2018`'s stations are correlated), or directories that record one
system on one clock (`bizitobs_application` and the services of that
application in `bizitobs_service`). Within a timeline every context of a call
ends where every label starts, so none reaches into a related series' label:
on the directories with real timestamps 0% of the label points are exposed
(`docs/validation/cohort_leakage.json`, the full test split, reproducible with
`docs/validation/check_cohort_leakage.py`). Different timelines in one call measure
unrelated things (Portuguese electricity meters, Beijing air quality, Seattle
loop detectors, Australian rain gauges); they do share calendar time, and 34.7%
of those label points fall inside some unrelated series' context period, which
the same file breaks down per directory. The M4 directories carry placeholder
timestamps (the competition series are anonymised), which order nothing, so
there the check is on content: 722 pairs of M4 series, where one holds the
other's test label verbatim (533 labels) or both are the same series stored
under two directories, are always put in different calls
(`make_cases.content_conflicts`). The test split has 82 calls, the 82 cutoffs of
`electricity`; `index.json` carries `n_cohorts` and each call's `cohort`. To
reproduce one call, keep the instances of one cohort with
`ts_io.write_case_subset`, as the verifier does. A package that loads its
weights once per call pays that cost up to 82 times;
budget for it: the 6.4M-parameter starter takes about 4 s of start-up plus
inference, 8.5 s per call on average and 42 s on the largest call, 12 minutes
over the full split (`docs/validation/task_trained_baseline.json`).

A worked example of both directories, with a valid output, is generated
without any real data by `python3 /workspace/evaluation/make_example.py --out /tmp/ex`.

`INPUT_DIR/index.json`:

```json
{"contract": "timeseries-forecast/v1", "split": "test", "context_cap": 16384, "n_cohorts": 82, "cohort": 0,
 "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
 "configs": [{"config_id": "electricity__W__short", "leaderboard_key": "electricity/W/short",
              "dataset": "electricity/W", "term": "short", "freq": "W-FRI", "prediction_length": 8,
              "seasonality": 1, "n_variates": 1, "n_instances": 370, "domain": "Energy"}, ...]}
```

`INPUT_DIR/cases/<config_id>.h5`, one file per (dataset, frequency, term)
config, holding every forecast instance of that config (`evaluation/ts_io.py`
has a reader, `CaseFile`):

| dataset / attr | content |
|---|---|
| `values` float32 (total,) | all contexts concatenated; instance `i` is `values[offsets[i]:offsets[i+1]]` reshaped to `(n_variates, context_length[i])` |
| `offsets` int64 (N+1,) | see above |
| `context_length` int64 (N,) | observations given per instance: the whole history before the cutoff, capped at the last `context_cap` = 16,384 |
| `history_length` int64 (N,) | length of the uncapped history (informational; it also reveals the series length and hence every test cutoff, which are public in GIFT-Eval anyway) |
| `cohort` int64 (N,), `source_index` int64 (N,) | the instance's cohort, and its index in the config's full instance list (informational: useful for caching or logging) |
| `item_id` str (N,), `window` int64 (N,) | GIFT-Eval item id and rolling-window index of the instance |
| `forecast_start` str (N,) | ISO timestamp of the first step to forecast |
| attrs | `dataset`, `freq` (pandas alias, e.g. `15T`, `H`, `W-SUN`), `freq_unit`, `term`, `prediction_length` H, `seasonality` m, `windows`, `n_variates` D, `domain`, `split`, `context_cap` |

Missing observations are `NaN` exactly where the benchmark has them; the
package must cope with them (impute, mask, or otherwise). Every instance is
self-contained: a package may not use one instance's context to inform another
(later windows of the same series contain earlier windows' labels; using them
is test-set leakage and is the reason each instance carries its own copy).
Parameters are fixed at inference: no test-time training or adaptation on the
contexts. An ensemble of models you trained counts as one package
when every member forecasts every instance.

## Output

`OUTPUT_DIR/cases/<config_id>.h5` for every config in `index.json`, a regular
HDF5 file whose `mean` and `quantiles` are ordinary datasets stored in it (as
`h5py`'s `create_dataset` or `ts_io.write_prediction_file` write them): a
symlink, an external or soft link, a virtual dataset or external raw storage
is not read and the config counts as failed. The layout
(`ts_io.write_prediction_file`):

| dataset | shape | meaning |
|---|---|---|
| `mean` | float32 (N, D, H) | point forecast (not scored; kept for compatibility with the leaderboard CSV columns) |
| `quantiles` | float32 (N, D, 9, H) | forecasts of the deciles 0.1 … 0.9 in that order; index 4 is the median used by MASE |

**Row order is the input's order.** Row `i` of `mean` and `quantiles` must be
the forecast for row `i` of that call's case file, and `N` must equal that
file's instance count. The verifier merges the calls positionally; it cannot
detect a permutation, so a package that reorders instances internally (for
example to batch by context length, as the starter does) must restore the
original order before writing.

plus `OUTPUT_DIR/model_manifest.json` (a copy of the package manifest; its
presence is recorded, not scored). All values must be finite. Quantiles
should be non-decreasing along the level axis; the grader sorts them
(monotone rearrangement) and reports the fraction of elements it had to
reorder. Inference must be deterministic: the verifier reruns one cohort and
records the largest difference in `status.json` (`determinism`); a
non-deterministic package is reported, not penalized.

## Scoring

Per config, exactly the two GIFT-Eval headline metrics as gluonts 0.15.1
computes them under `evaluate_model(..., axis=None, mask_invalid_label=True)`
(`evaluation/score.py`; checked against gluonts on the same forecasts to
1.9e-7 on all 97 configs and against the leaderboard's Seasonal Naive CSV,
`docs/validation/`):

- **MASE** = mean over all (instance, variate, step) elements with a finite
  label and a positive seasonal error of |y − q̂₀.₅| / seasonal_error, where
  seasonal_error is the mean absolute m-lag difference of the instance's full
  (uncapped) history over the pairs where both values are finite (m = the
  frequency's seasonality; lag 1 if m exceeds the history).
- **WQL** (the leaderboard column `mean_weighted_sum_quantile_loss`, labelled
  CRPS on the board) = mean over the nine levels q of
  Σ 2|(y − q̂_q)(1{q̂_q ≥ y} − q)| / Σ |y|, sums over the finite-label elements.

Each is divided by the Seasonal Naive value for the same config taken from the
leaderboard's own `Seasonal_Naive` row at the pinned Space revision
(`evaluation/seasonal_naive_reference.json`; the task's contract-compliant
Seasonal Naive package, `baseline/seasonal_naive`, reproduces those values
except where the context cap changes its interval width,
`docs/validation/seasonal_naive_replay.json`), clipped to [1e-3, 10], and aggregated
with a geometric mean over the 97 configs, which is how the leaderboard
aggregates. Every config weighs the same, so the 21 dataset-frequency variants
that have all three terms carry three times the weight of the other 34, as on
the board. Then

    skill  = 1 − sqrt(gmean(rel MASE) × gmean(rel WQL))
    reward = clip(skill, 0, 1)

A reward of 0 means no better than the leaderboard's Seasonal Naive row on the
geometric average, 1 would be a perfect forecast; the unclipped `skill_score`
is reported alongside so that progress below that floor stays visible. The
task's own contract-compliant Seasonal Naive scores 0.008 rather than exactly
0, because the context cap narrows its prediction intervals on three
jena_weather configs (`docs/validation/seasonal_naive_replay.json`); that offset
applies to every package equally. For orientation on the leaderboard's own
per-config numbers (relative MASE / relative WQL → reward): Naive 1.27 / 1.59 →
0, PatchTST per-dataset supervised 0.849 / 0.587 → 0.294, chronos_bolt_base
0.808 / 0.574 → 0.319, TimesFM-2.5 0.705 / 0.490 → 0.412, TimesFM-3 0.667 /
0.456 → 0.449. The report also carries the leaderboard's breakdowns (term,
domain, frequency, uni/multivariate) and a per-config CSV in the leaderboard's
column names, so any submission can be placed on the public board directly.

A config whose prediction file is missing, unreadable, mis-shaped or
non-finite scores the clip maximum (10) on both metrics; it never aborts the
run. A package that produces nothing gets reward 0. If the wall-clock budget
runs out part way, the configs already forecast keep their score and each
partly forecast config is scored on the instances it covers and blended toward
the clip maximum in proportion to what is missing, so finishing 80% of the
work is worth roughly 80% of the way from the failure value to the real score
rather than nothing.

## Deviations from GIFT-Eval

**Targets only.** GIFT-Eval carries `past_feat_dynamic_real` on 10
dataset-frequency variants (24 of the 97 configs); the case files here contain
the target history alone, so every package forecasts from the same
information. Replays of models that can use covariates are therefore slightly
below their board numbers on those configs and identical on the rest
(`docs/validation/timesfm3_zero_shot.json`).



**Context cap.** GIFT-Eval hands each model the entire history before the cutoff. Because every
instance here carries its own context, the history is capped at the last
16,384 observations (7.1 GB of staged input instead of about 28 GB). The seasonal
error is still computed on the uncapped history, so MASE is unaffected. The
cap exceeds the context used by every leaderboard model we compare against
(TimesFM-3 15,360; TimesFM-2.5 15,360; Chronos-2 8,192; Chronos-Bolt 2,048);
only a method that
would fit on longer histories (a statistical model on the 140k-point
15-minute electricity series, say) sees a difference.

## Rules recap

- Training data: the allow-listed `gifteval_pretrain` directories mounted at
  `/workspace/data/gifteval_pretrain` and synthetic series generated in the environment
  by the package author's own code. No other data; no pretrained weights of
  any kind; random initialization only.
- The validation cases under `/workspace/data/valid` (the window before each series'
  test region) are for model selection only, never for training.
- The test cases are built by the verifier from data the agent never sees.
- Inference uses only `INPUT_DIR`; no dataset files, no network (enforced by
  the namespace, not by trust).
- A manifest that declares `pretrained_assets` scores 0 (`status.json`
  records `rule_violation`); undeclared use is a rule violation that the
  verifier cannot detect and that a reviewer should look for in the package.

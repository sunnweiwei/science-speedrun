# Evaluation

| file | purpose |
|---|---|
| `ts_io.py` | protocol tables (horizons, seasonalities, window rule), pyarrow readers for the Arrow layout, the case / truth / prediction file formats, gluonts' seasonal error |
| `make_cases.py` | stages `INPUT_DIR` (per-instance contexts, capped at 16,384 observations) and `TRUTH_DIR` (labels + seasonal errors on the uncapped history) for `--split test` or `valid` |
| `score.py` | the grader: per-config MASE and WQL exactly as gluonts 0.15.1 computes them, relative to Seasonal Naive, geometric mean, reward; per-config CSV in the leaderboard's column names |
| `sync_readers.sh` (task repository only) | copies `ts_io.py` into the packages that must be self-contained (a unit test fails if a copy drifts) |
| `seasonal_naive_reference.json` (verifier only) | the normalizer: per-config MASE / WQL of the leaderboard's `Seasonal_Naive` row at the pinned Space revision (not produced by this grader; the task's own Seasonal Naive replay is compared with it in `docs/validation/seasonal_naive_replay.json`) |
| `seasonal_naive_reference_valid.json` | the validation-split reference, produced by `baseline/seasonal_naive` through this grader on the validation windows (a different scale from the test normalizer) |

Run it yourself on the validation windows (the only labels an agent has):

```
python3 score.py --truth /workspace/data/valid/truth --output <your predictions> \
    --baseline seasonal_naive_reference_valid.json --report /workspace/research/report.json --csv /workspace/research/results.csv
```

`--baseline` selects the validation-split Seasonal Naive reference; without it the grader uses the
bundled test-split normalizer, which is a different scale. The report carries per-config metrics and
the same breakdowns the verifier writes.

Validation of the grader (in the task repository's `docs/validation/`): `golden_gluonts_check.py` re-evaluates the Seasonal
Naive prediction files with gluonts + gift-eval and agrees with `score.py` to 1.9e-7 on all 97 configs
(the check covers the metric path; failure handling and quantile sorting are covered by unit tests);
the Seasonal Naive replay through the contract reproduces the leaderboard's per-config MASE on all
97 configs and WQL on 63; the TimesFM-2.5 replay reproduces its published per-config CSV to 4.3e-6 and the Chronos-2 replay to 0.04%.

Reward: `clip(1 − sqrt(gmean(rel MASE) × gmean(rel WQL)), 0, 1)`. Failed configs (missing,
unreadable, mis-shaped or non-finite predictions) count as relative error 10 on both metrics,
so a package that forecasts nothing gets 0 and one that fails a single config loses about 0.016
reward relative to a config it would otherwise have scored at the Chronos-2 level.

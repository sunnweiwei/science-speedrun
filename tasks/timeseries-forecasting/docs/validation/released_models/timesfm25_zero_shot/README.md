# timesfm25_zero_shot — external-target replay of TimesFM-2.5

Replays the GIFT-Eval leaderboard row **TimesFM-2.5** under the task contract
`timeseries-forecast/v1`. Nothing is trained: the released checkpoint is downloaded at
run time and evaluated zero-shot.

## What it replays

| | |
| --- | --- |
| leaderboard row | `TimesFM-2.5` (`results/TimesFM-2.5/all_results.csv` of the GIFT-Eval space) |
| published aggregate | 0.7050 MASE / 0.4903 WQL (geometric mean over the 97 configs of model / Seasonal_Naive) |
| official recipe | `gift-eval/notebooks/timesfm2p5.ipynb` |
| checkpoint | `google/timesfm-2.5-200m-pytorch` @ `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| checkpoint license | Apache-2.0 (weights are still not copied into this repo) |
| parameters | 231,289,280 |
| code | `timesfm` 3.0.2 — `github.com/google-research/timesfm` @ `e31dadd84cb26bd5153fde6687502b8312e918fb` |

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate     # python 3.11
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install numpy h5py huggingface_hub \
  "timesfm[torch] @ git+https://github.com/google-research/timesfm@e31dadd84cb26bd5153fde6687502b8312e918fb"

bash infer.sh INPUT_DIR OUTPUT_DIR
```

`predict.py` downloads the checkpoint at the pinned revision into
`$TIMESERIES_PRETRAINED_DIR/timesfm-2.5-200m-pytorch` (default `/data/pretrained`)
on first use, or reuses `$TIMESFM25_CHECKPOINT`. Partial outputs are resumable:
a config whose `cases/<config_id>.h5` already exists is skipped
(`TIMESERIES_RESUME=0` to force a rerun).

## Settings (from the official notebook)

`timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch`, recompiled once
per 1024-series chunk exactly as the notebook's `TimesFmPredictor.predict` does:

```python
timesfm.configs.ForecastConfig(
    max_context=min(15360, patch_round_up(max context length in the chunk)),
    max_horizon=1024, infer_is_positive=True, use_continuous_quantile_head=True,
    fix_quantile_crossing=True, force_flip_invariance=True, return_backcast=False,
    normalize_inputs=True, per_core_batch_size=...)
```

The nine deciles are `full_preds[:, :H, 1:]` — channel 0 of the model's quantile
output is its mean head, channels 1..9 the deciles 0.1..0.9 — which is precisely
what the notebook feeds to `QuantileForecast`. **Multivariate configs are
flattened into independent univariate series**, the notebook's
`to_univariate=True` branch (TimesFM-2.5 is a univariate model). NaNs are passed
to the model as NaN: `TimesFM_2p5.forecast` strips leading NaNs and linearly
interpolates the rest, as in the notebook.

The prediction file's `mean` array holds the model's mean channel (index 0). The
leaderboard's `MSE[mean]` was instead computed from the median, because the
notebook builds a `QuantileForecast` without a mean; this makes no difference to
the two metrics the task grades (MASE uses q0.5, WQL uses the nine deciles).

## Deviations from the notebook

1. **Context length.** The task's case files cap history at 16384 observations
   (GIFT-Eval passes the full history); `max_context` is capped at 15360 by the
   notebook anyway, so this is a no-op for every config whose history exceeds
   15360.
2. **`per_core_batch_size`.** The notebook uses 128; here it is
   `min(128, TIMESFM25_TOKEN_BUDGET / max_context)` (budget 196608 by default,
   i.e. 12 series at a 15360-step context) so the run fits in ~3 GB of a shared
   GPU. `TimesFM_2p5.forecast` pads and masks every series to `max_context`
   independently of the batch size and the leftover slots are discarded, so this
   is pure chunking and provably cannot change a forecast.
3. **`max_horizon` is rounded up from the config's horizon instead of fixed at
   1024.** `TimesFM_2p5.decode` is autoregressive over output patches of 128
   steps, and the continuous-quantile head always reads a 1024-step quantile
   spread off the last context patch, so the first H forecast steps do not
   depend on `max_horizon`. Using `ceil(H/128)*128` skips up to 7 autoregressive
   patches per series and is **bit-identical**: verified on the smoke split,
   where `max_horizon=1024` and `max_horizon=auto` produce byte-for-byte equal
   `mean` and `quantiles` arrays for all 7 configs while the run is 5.3x faster
   (257.7 s -> 48.4 s). Set `TIMESFM25_MAX_HORIZON=1024` for the notebook's
   literal setting.
4. **`torch_compile=False`.** The notebook leaves `TimesFM_2p5_200M_torch`'s
   default `torch_compile=True`, but the notebook also recompiles the forecast
   config for every chunk, so every new (batch, context) shape triggers a fresh
   `torch.compile` trace. With 97 configs and 371,330 series the tracing cost
   dominates, so eager mode is used instead. Set
   `TIMESFM25_TORCH_COMPILE=1` to restore the notebook's setting.
5. **All-NaN contexts.** `TimesFM_2p5.forecast` crashes on a context with no
   finite value (`linear_interpolation` guards with `if non_nans_values:`, which
   raises on an empty numpy array). 649 of the 371,330 univariate series of the
   test split are entirely NaN inside the 16384-step window — every
   `bitbrains_*` config plus one `electricity/W` series. `predict.py` replaces
   such a series with zeros, which is what TimesFM-3's own forecaster does with
   the same input. Their labels are NaN and masked by the grader anyway.

## Result on the task's full test split (97 configs, 326,490 instances)

| | ours | published |
| --- | ---: | ---: |
| geometric mean of MASE / Seasonal_Naive over the 97 configs | **0.7050** | 0.7050 |
| geometric mean of WQL / Seasonal_Naive over the 97 configs | **0.4903** | 0.4903 |
| task grader reward (skill over the task's Seasonal Naive) | 0.4121 | — |
| configs failed | 0 | — |

Per-config agreement with `results/TimesFM-2.5/all_results.csv`: **maximum
relative deviation 1.6e-6 (MASE) and 4.3e-6 (WQL)**, median 0.0 for both, 93 of
97 configs agree to within 1e-6 on both metrics, and **no config deviates by
more than 5%** (none by more than 0.001%). The remaining 1e-6-level differences
are float32 GPU non-determinism.

Runtime: 55,025 s of model time summed over the 97 configs, of which ~20,500 s
is an externally imposed SIGSTOP pause inside `electricity/15T/medium`, i.e.
**~9.6 h net** on one A100 40GB shared with other jobs, at
`TIMESFM25_TOKEN_BUDGET=122880` (per-core batch 8 at a 15360-step context);
peak GPU memory ~3.0 GB, peak host RSS 2.5 GB.

Artefacts of that run: predictions
`/large-disk/tsfm/work/test/output_timesfm25`, grader report
`.../report_timesfm25.json`, comparison `.../compare_timesfm25.json`.

### Per-config comparison

| config | MASE ours | MASE published | dev | WQL ours | WQL published | dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bitbrains_fast_storage/5T/long | 0.9039 | 0.9039 | +0.00% | 0.8315 | 0.8315 | +0.00% |
| bitbrains_fast_storage/5T/medium | 0.9945 | 0.9945 | +0.00% | 0.7661 | 0.7661 | -0.00% |
| bitbrains_fast_storage/5T/short | 0.7111 | 0.7111 | +0.00% | 0.3973 | 0.3973 | -0.00% |
| bitbrains_fast_storage/H/short | 1.0595 | 1.0595 | +0.00% | 0.7824 | 0.7824 | +0.00% |
| bitbrains_rnd/5T/long | 3.3142 | 3.3142 | -0.00% | 0.7435 | 0.7435 | +0.00% |
| bitbrains_rnd/5T/medium | 4.3935 | 4.3935 | -0.00% | 0.8462 | 0.8462 | +0.00% |
| bitbrains_rnd/5T/short | 1.6571 | 1.6571 | -0.00% | 0.4079 | 0.4079 | +0.00% |
| bitbrains_rnd/H/short | 5.8512 | 5.8512 | -0.00% | 0.6100 | 0.6100 | -0.00% |
| bizitobs_application/10S/long | 3.1622 | 3.1622 | +0.00% | 0.0534 | 0.0534 | +0.00% |
| bizitobs_application/10S/medium | 2.1552 | 2.1552 | -0.00% | 0.0326 | 0.0326 | -0.00% |
| bizitobs_application/10S/short | 1.0241 | 1.0241 | +0.00% | 0.0095 | 0.0095 | +0.00% |
| bizitobs_l2c/5T/long | 0.6111 | 0.6111 | +0.00% | 0.2791 | 0.2791 | +0.00% |
| bizitobs_l2c/5T/medium | 0.5704 | 0.5704 | +0.00% | 0.2409 | 0.2409 | -0.00% |
| bizitobs_l2c/5T/short | 0.2741 | 0.2741 | +0.00% | 0.0721 | 0.0721 | -0.00% |
| bizitobs_l2c/H/long | 0.5939 | 0.5939 | -0.00% | 0.2733 | 0.2733 | -0.00% |
| bizitobs_l2c/H/medium | 0.5056 | 0.5056 | -0.00% | 0.2372 | 0.2372 | -0.00% |
| bizitobs_l2c/H/short | 0.4261 | 0.4261 | -0.00% | 0.1795 | 0.1795 | +0.00% |
| bizitobs_service/10S/long | 1.3104 | 1.3104 | +0.00% | 0.0505 | 0.0505 | -0.00% |
| bizitobs_service/10S/medium | 0.9591 | 0.9591 | -0.00% | 0.0177 | 0.0177 | -0.00% |
| bizitobs_service/10S/short | 0.7261 | 0.7261 | +0.00% | 0.0102 | 0.0102 | +0.00% |
| car_parts/M/short | 0.8384 | 0.8384 | +0.00% | 0.9421 | 0.9421 | +0.00% |
| covid_deaths/D/short | 36.9065 | 36.9065 | +0.00% | 0.0352 | 0.0352 | +0.00% |
| electricity/15T/long | 0.9183 | 0.9183 | -0.00% | 0.0774 | 0.0774 | -0.00% |
| electricity/15T/medium | 0.8657 | 0.8657 | -0.00% | 0.0766 | 0.0766 | -0.00% |
| electricity/15T/short | 1.0814 | 1.0814 | +0.00% | 0.0954 | 0.0954 | -0.00% |
| electricity/D/short | 1.3764 | 1.3764 | -0.00% | 0.0541 | 0.0541 | +0.00% |
| electricity/H/long | 1.1440 | 1.1440 | +0.00% | 0.0817 | 0.0817 | -0.00% |
| electricity/H/medium | 1.0555 | 1.0555 | -0.00% | 0.0733 | 0.0733 | -0.00% |
| electricity/H/short | 0.9075 | 0.9075 | +0.00% | 0.0646 | 0.0646 | -0.00% |
| electricity/W/short | 1.3968 | 1.3968 | +0.00% | 0.0461 | 0.0461 | +0.00% |
| ett1/15T/long | 1.0596 | 1.0596 | -0.00% | 0.2555 | 0.2555 | -0.00% |
| ett1/15T/medium | 1.0375 | 1.0375 | -0.00% | 0.2512 | 0.2512 | -0.00% |
| ett1/15T/short | 0.6939 | 0.6939 | +0.00% | 0.1569 | 0.1569 | -0.00% |
| ett1/D/short | 1.7128 | 1.7128 | +0.00% | 0.2996 | 0.2996 | +0.00% |
| ett1/H/long | 1.4466 | 1.4466 | -0.00% | 0.2878 | 0.2878 | -0.00% |
| ett1/H/medium | 1.3476 | 1.3476 | -0.00% | 0.2867 | 0.2867 | -0.00% |
| ett1/H/short | 0.8387 | 0.8387 | -0.00% | 0.1880 | 0.1880 | -0.00% |
| ett1/W/short | 1.4441 | 1.4441 | +0.00% | 0.2405 | 0.2405 | +0.00% |
| ett2/15T/long | 0.9372 | 0.9372 | -0.00% | 0.0967 | 0.0967 | -0.00% |
| ett2/15T/medium | 0.9149 | 0.9149 | +0.00% | 0.0940 | 0.0940 | -0.00% |
| ett2/15T/short | 0.7237 | 0.7237 | +0.00% | 0.0628 | 0.0628 | +0.00% |
| ett2/D/short | 1.2743 | 1.2743 | +0.00% | 0.0920 | 0.0920 | +0.00% |
| ett2/H/long | 1.0070 | 1.0070 | +0.00% | 0.1007 | 0.1007 | +0.00% |
| ett2/H/medium | 1.0082 | 1.0082 | -0.00% | 0.1006 | 0.1006 | -0.00% |
| ett2/H/short | 0.7392 | 0.7392 | +0.00% | 0.0645 | 0.0645 | +0.00% |
| ett2/W/short | 0.8412 | 0.8412 | -0.00% | 0.0869 | 0.0869 | -0.00% |
| hierarchical_sales/D/short | 0.7450 | 0.7450 | +0.00% | 0.5736 | 0.5736 | +0.00% |
| hierarchical_sales/W/short | 0.7191 | 0.7191 | +0.00% | 0.3478 | 0.3478 | +0.00% |
| hospital/M/short | 0.7590 | 0.7590 | +0.00% | 0.0507 | 0.0507 | +0.00% |
| jena_weather/10T/long | 0.6750 | 0.6750 | -0.00% | 0.0499 | 0.0499 | +0.00% |
| jena_weather/10T/medium | 0.6271 | 0.6271 | +0.00% | 0.0495 | 0.0495 | +0.00% |
| jena_weather/10T/short | 0.2760 | 0.2760 | +0.00% | 0.0283 | 0.0283 | -0.00% |
| jena_weather/D/short | 0.9881 | 0.9881 | -0.00% | 0.0445 | 0.0445 | -0.00% |
| jena_weather/H/long | 0.8487 | 0.8487 | +0.00% | 0.0551 | 0.0551 | +0.00% |
| jena_weather/H/medium | 0.7295 | 0.7295 | -0.00% | 0.0507 | 0.0507 | -0.00% |
| jena_weather/H/short | 0.5257 | 0.5257 | -0.00% | 0.0434 | 0.0434 | +0.00% |
| kdd_cup_2018/D/short | 1.1915 | 1.1915 | +0.00% | 0.3752 | 0.3752 | +0.00% |
| kdd_cup_2018/H/long | 1.0078 | 1.0078 | +0.00% | 0.4399 | 0.4399 | +0.00% |
| kdd_cup_2018/H/medium | 1.0358 | 1.0358 | -0.00% | 0.4190 | 0.4190 | +0.00% |
| kdd_cup_2018/H/short | 0.9326 | 0.9326 | -0.00% | 0.3739 | 0.3739 | -0.00% |
| loop_seattle/5T/long | 0.8448 | 0.8448 | +0.00% | 0.0759 | 0.0759 | +0.00% |
| loop_seattle/5T/medium | 0.7873 | 0.7873 | +0.00% | 0.0709 | 0.0709 | +0.00% |
| loop_seattle/5T/short | 0.5676 | 0.5676 | +0.00% | 0.0491 | 0.0491 | +0.00% |
| loop_seattle/D/short | 0.8627 | 0.8627 | -0.00% | 0.0409 | 0.0409 | -0.00% |
| loop_seattle/H/long | 0.8222 | 0.8222 | -0.00% | 0.0570 | 0.0570 | -0.00% |
| loop_seattle/H/medium | 0.7994 | 0.7994 | -0.00% | 0.0554 | 0.0554 | +0.00% |
| loop_seattle/H/short | 0.7250 | 0.7250 | +0.00% | 0.0508 | 0.0508 | +0.00% |
| m4_daily/D/short | 3.2974 | 3.2974 | -0.00% | 0.0217 | 0.0217 | +0.00% |
| m4_hourly/H/short | 0.7336 | 0.7336 | -0.00% | 0.0236 | 0.0236 | +0.00% |
| m4_monthly/M/short | 0.9462 | 0.9462 | -0.00% | 0.0940 | 0.0940 | -0.00% |
| m4_quarterly/Q/short | 1.2050 | 1.2050 | -0.00% | 0.0761 | 0.0761 | -0.00% |
| m4_weekly/W/short | 2.0020 | 2.0020 | -0.00% | 0.0367 | 0.0367 | +0.00% |
| m4_yearly/A/short | 3.5746 | 3.5746 | -0.00% | 0.1271 | 0.1271 | -0.00% |
| m_dense/D/short | 0.6615 | 0.6615 | +0.00% | 0.0626 | 0.0626 | +0.00% |
| m_dense/H/long | 0.6619 | 0.6619 | -0.00% | 0.1125 | 0.1125 | -0.00% |
| m_dense/H/medium | 0.6637 | 0.6637 | -0.00% | 0.1125 | 0.1125 | -0.00% |
| m_dense/H/short | 0.7688 | 0.7688 | -0.00% | 0.1277 | 0.1277 | +0.00% |
| restaurant/D/short | 0.6818 | 0.6818 | -0.00% | 0.2565 | 0.2565 | -0.00% |
| saugeen/D/short | 2.7150 | 2.7150 | -0.00% | 0.3334 | 0.3334 | -0.00% |
| saugeen/M/short | 0.7144 | 0.7144 | -0.00% | 0.2980 | 0.2980 | +0.00% |
| saugeen/W/short | 1.2478 | 1.2478 | -0.00% | 0.3781 | 0.3781 | -0.00% |
| solar/10T/long | 0.8528 | 0.8528 | -0.00% | 0.3309 | 0.3309 | +0.00% |
| solar/10T/medium | 0.8420 | 0.8420 | -0.00% | 0.3404 | 0.3404 | +0.00% |
| solar/10T/short | 1.0857 | 1.0857 | +0.00% | 0.5624 | 0.5624 | +0.00% |
| solar/D/short | 0.9852 | 0.9852 | -0.00% | 0.2821 | 0.2821 | +0.00% |
| solar/H/long | 0.9437 | 0.9437 | -0.00% | 0.3245 | 0.3245 | -0.00% |
| solar/H/medium | 0.8790 | 0.8790 | -0.00% | 0.3145 | 0.3145 | -0.00% |
| solar/H/short | 0.8976 | 0.8976 | -0.00% | 0.3368 | 0.3368 | -0.00% |
| solar/W/short | 1.2792 | 1.2792 | +0.00% | 0.1705 | 0.1705 | +0.00% |
| sz_taxi/15T/long | 0.5131 | 0.5131 | +0.00% | 0.1998 | 0.1998 | +0.00% |
| sz_taxi/15T/medium | 0.5393 | 0.5393 | -0.00% | 0.2032 | 0.2032 | -0.00% |
| sz_taxi/15T/short | 0.5444 | 0.5444 | +0.00% | 0.2006 | 0.2006 | +0.00% |
| sz_taxi/H/short | 0.5633 | 0.5633 | -0.00% | 0.1357 | 0.1357 | -0.00% |
| temperature_rain/D/short | 1.3471 | 1.3471 | +0.00% | 0.5531 | 0.5531 | +0.00% |
| us_births/D/short | 0.3430 | 0.3430 | -0.00% | 0.0182 | 0.0182 | -0.00% |
| us_births/M/short | 0.6083 | 0.6083 | +0.00% | 0.0126 | 0.0126 | -0.00% |
| us_births/W/short | 1.0866 | 1.0866 | +0.00% | 0.0128 | 0.0128 | +0.00% |

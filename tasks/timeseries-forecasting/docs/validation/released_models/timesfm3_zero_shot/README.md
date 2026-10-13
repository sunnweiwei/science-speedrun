# timesfm3_zero_shot — external-target replay of TimesFM-3

Replays the GIFT-Eval leaderboard row **TimesFM-3** under the task contract
`timeseries-forecast/v1`. Nothing is trained: the released checkpoint is downloaded at
run time and evaluated zero-shot.

## What it replays

| | |
| --- | --- |
| leaderboard row | `TimesFM-3` (`results/TimesFM-3/all_results.csv` of the GIFT-Eval space) |
| published aggregate | 0.6668 MASE / 0.4557 WQL (geometric mean over the 97 configs of model / Seasonal_Naive) |
| official recipe | `gift-eval/notebooks/timesfm3.ipynb` |
| checkpoint | `google/timesfm-3.0-pytorch` @ `43046b85ec22d584a13f8098c2ed39c889e129c2` |
| checkpoint license | `timesfm-non-commercial-license-v1.0` (**non-commercial**; weights are never copied into this repo) |
| parameters | 330,710,976 |
| code | `timesfm` 3.0.2 — `github.com/google-research/timesfm` @ `e31dadd84cb26bd5153fde6687502b8312e918fb` (package `timesfm3`) |

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate     # python 3.11
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install numpy h5py huggingface_hub \
  "timesfm[torch] @ git+https://github.com/google-research/timesfm@e31dadd84cb26bd5153fde6687502b8312e918fb"

bash infer.sh INPUT_DIR OUTPUT_DIR
```

`predict.py` downloads the checkpoint at the pinned revision into
`$TIMESERIES_PRETRAINED_DIR/timesfm-3.0-pytorch` (default `/data/pretrained`) on
first use, or reuses `$TIMESFM3_CHECKPOINT`. Partial outputs are resumable:
a config whose `cases/<config_id>.h5` already exists is skipped
(`TIMESERIES_RESUME=0` to force a rerun).

## Settings (from the official notebook)

`timesfm3.TimesFM3Evaluator(ModelConfig(input_patch_length=32,
output_patch_length=64, per_core_batch_size=64, use_variate_attention=True,
use_sdpa=True))`, called as

```python
forecaster.predict_batch(contexts=..., horizon=H, past_only_covariates=[None]*n,
                         return_quantiles=True, use_symmetric_averaging=True,
                         make_positive=True, sort_quantiles=True)
```

with the notebook's adaptive batch size — 16 for a multivariate config, 64 for a
univariate one. Univariate configs are passed as 1-D arrays and multivariate
configs as `(D, T)` arrays, i.e. **the notebook's `to_univariate=False`**:
TimesFM-3 forecasts all variates of an instance in one forward pass through its
cross-variate attention (at most 32 variates per pass, chunked by the
evaluator). NaNs are passed to the model as NaN — the forecaster strips leading
all-NaN steps and linearly interpolates the rest, which is exactly what the
notebook's inputs go through. The model truncates every context to its own
15360-step limit. The nine deciles come from the model's quantile head, already
monotone (`sort_quantiles=True`). The `mean` array of the prediction file is the
model's median quantile: TimesFM-3 has no separate mean head, and the notebook's
`QuantileForecast` also falls back to q0.5 for `MSE[mean]`.

## Deviations from the notebook

1. **Context length.** The task's case files cap history at 16384 observations
   (GIFT-Eval passes the full history); the model truncates to its own 15360
   limit anyway, so this is a no-op for every config whose history exceeds 15360.
2. **Inner batch size under a GPU-memory budget.** The outer batch stays at the
   notebook's 64 / 16, but the evaluator's `per_core_batch_size` is additionally
   capped at `TIMESFM3_TOKEN_BUDGET / (context_length * variates)` (default
   245760, i.e. 16 series at a 15360-step context) so the run fits in ~2 GB of a
   shared GPU. This only regroups the forward passes; padded steps are masked,
   so it does not change the forecasts (confirmed on the smoke split, where
   every config runs at the notebook's batch size and reproduces the leaderboard
   to 4 decimals).
3. **No past covariates.** The notebook forwards `past_feat_dynamic_real` when a
   GIFT-Eval entry has it; the task's case files carry targets only, so
   `past_only_covariates` is always `None`. 24 of the 97 configs are affected —
   the `M_DENSE`, `bitbrains_fast_storage`, `bitbrains_rnd`,
   `bizitobs_application`, `bizitobs_l2c` and `bizitobs_service` configs — and
   this is the one deviation that can move a number. It is the expected source
   of any residual gap on those configs in the table below.

## Result on the task's full test split (97 configs, 326,490 instances)

| | ours | published |
| --- | ---: | ---: |
| geometric mean of MASE / Seasonal_Naive over the 97 configs | 0.6685 | 0.6668 |
| geometric mean of WQL / Seasonal_Naive over the 97 configs | 0.4583 | 0.4557 |
| the same over the **73 configs with no past covariates** | **0.7120 / 0.4796** | 0.7120 / 0.4796 |
| task grader reward (skill over the task's Seasonal Naive) | 0.4465 | — |
| configs failed | 0 | — |

The split by covariates is the whole story of the residual gap:

| group | configs | max rel. dev. MASE | median | max rel. dev. WQL | median | agree to 1e-6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no `past_feat_dynamic_real` | 73 | 0.199% | 0.000% | 0.213% | 0.000% | 56 |
| has `past_feat_dynamic_real` | 24 | 6.944% | 0.743% | 14.915% | 1.128% | 0 |

On the 73 configs whose GIFT-Eval entries carry no past covariates the replay is
**exact** — the aggregate reproduces the leaderboard to four decimals and 56 of
them agree to within 1e-6 on both metrics, the rest to within 0.2% (float32 GPU
non-determinism and the 16384 vs full-history context). All five configs that
deviate by more than 5% carry past covariates that the task's case files do not
provide:

| config | MASE dev | WQL dev |
| --- | ---: | ---: |
| bitbrains_fast_storage/5T/short | +0.68% | +7.97% |
| bitbrains_fast_storage/5T/medium | +2.72% | +14.91% |
| bitbrains_fast_storage/5T/long | +4.35% | +5.66% |
| bitbrains_fast_storage/H/short | +1.58% | +11.29% |
| bizitobs_application/10S/short | +6.94% | +5.22% |

Runtime: 35,161 s of model time summed over the 97 configs (**9.8 h**) on one
A100 40GB shared with other jobs, at `TIMESFM3_TOKEN_BUDGET=245760` (inner
per-core batch 16 at a 15360-step context, the notebook's 64 / 16 everywhere
else); peak GPU memory ~2.7 GB, peak host RSS 3.1 GB.

Artefacts of that run: predictions
`/large-disk/tsfm/work/test/output_timesfm3`, grader report
`.../report_timesfm3.json`, comparison `.../compare_timesfm3.json`.

### Per-config comparison

| config | MASE ours | MASE published | dev | WQL ours | WQL published | dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bitbrains_fast_storage/5T/long | 0.8954 | 0.8581 | +4.35% | 0.7031 | 0.6654 | +5.66% |
| bitbrains_fast_storage/5T/medium | 0.9690 | 0.9434 | +2.72% | 0.6742 | 0.5867 | +14.91% |
| bitbrains_fast_storage/5T/short | 0.6552 | 0.6508 | +0.68% | 0.3898 | 0.3610 | +7.97% |
| bitbrains_fast_storage/H/short | 1.0027 | 0.9871 | +1.58% | 0.7209 | 0.6478 | +11.29% |
| bitbrains_rnd/5T/long | 3.2755 | 3.2660 | +0.29% | 0.6192 | 0.6185 | +0.12% |
| bitbrains_rnd/5T/medium | 4.3500 | 4.3392 | +0.25% | 0.6272 | 0.6107 | +2.70% |
| bitbrains_rnd/5T/short | 1.6052 | 1.6006 | +0.28% | 0.3742 | 0.3643 | +2.72% |
| bitbrains_rnd/H/short | 5.8218 | 5.8249 | -0.05% | 0.6115 | 0.6046 | +1.13% |
| bizitobs_application/10S/long | 3.0553 | 3.0348 | +0.67% | 0.0478 | 0.0487 | -2.02% |
| bizitobs_application/10S/medium | 2.1470 | 2.0907 | +2.69% | 0.0318 | 0.0321 | -0.91% |
| bizitobs_application/10S/short | 0.8945 | 0.8364 | +6.94% | 0.0095 | 0.0090 | +5.22% |
| bizitobs_l2c/5T/long | 0.4644 | 0.4744 | -2.11% | 0.2055 | 0.2078 | -1.13% |
| bizitobs_l2c/5T/medium | 0.4781 | 0.4805 | -0.52% | 0.1981 | 0.1969 | +0.62% |
| bizitobs_l2c/5T/short | 0.2282 | 0.2241 | +1.83% | 0.0625 | 0.0615 | +1.60% |
| bizitobs_l2c/H/long | 0.5439 | 0.5486 | -0.85% | 0.2495 | 0.2513 | -0.73% |
| bizitobs_l2c/H/medium | 0.4802 | 0.4793 | +0.18% | 0.2260 | 0.2259 | +0.07% |
| bizitobs_l2c/H/short | 0.3922 | 0.3907 | +0.38% | 0.1677 | 0.1674 | +0.20% |
| bizitobs_service/10S/long | 1.2566 | 1.2494 | +0.58% | 0.0501 | 0.0503 | -0.31% |
| bizitobs_service/10S/medium | 0.9430 | 0.9354 | +0.81% | 0.0184 | 0.0183 | +0.15% |
| bizitobs_service/10S/short | 0.6648 | 0.6754 | -1.57% | 0.0100 | 0.0100 | +0.10% |
| car_parts/M/short | 0.8446 | 0.8446 | +0.00% | 0.9654 | 0.9654 | +0.00% |
| covid_deaths/D/short | 31.8585 | 31.8585 | -0.00% | 0.0288 | 0.0288 | -0.00% |
| electricity/15T/long | 0.8632 | 0.8632 | -0.00% | 0.0723 | 0.0723 | -0.00% |
| electricity/15T/medium | 0.8178 | 0.8178 | +0.00% | 0.0724 | 0.0724 | -0.00% |
| electricity/15T/short | 0.9932 | 0.9932 | -0.00% | 0.0857 | 0.0857 | -0.00% |
| electricity/D/short | 1.3508 | 1.3508 | +0.00% | 0.0530 | 0.0530 | -0.00% |
| electricity/H/long | 1.0756 | 1.0756 | -0.00% | 0.0743 | 0.0743 | +0.00% |
| electricity/H/medium | 1.0030 | 1.0030 | -0.01% | 0.0695 | 0.0695 | -0.00% |
| electricity/H/short | 0.8533 | 0.8533 | +0.00% | 0.0584 | 0.0584 | +0.00% |
| electricity/W/short | 1.3947 | 1.3947 | +0.00% | 0.0478 | 0.0478 | +0.00% |
| ett1/15T/long | 0.9751 | 0.9751 | +0.00% | 0.2227 | 0.2227 | -0.00% |
| ett1/15T/medium | 0.9673 | 0.9673 | -0.00% | 0.2237 | 0.2237 | +0.00% |
| ett1/15T/short | 0.6612 | 0.6612 | -0.00% | 0.1518 | 0.1518 | -0.00% |
| ett1/D/short | 1.6757 | 1.6757 | +0.00% | 0.2710 | 0.2710 | +0.00% |
| ett1/H/long | 1.3952 | 1.3951 | +0.00% | 0.2727 | 0.2727 | +0.00% |
| ett1/H/medium | 1.2695 | 1.2695 | +0.00% | 0.2556 | 0.2556 | +0.00% |
| ett1/H/short | 0.8157 | 0.8157 | +0.00% | 0.1766 | 0.1766 | +0.00% |
| ett1/W/short | 1.4723 | 1.4723 | -0.00% | 0.2496 | 0.2496 | +0.00% |
| ett2/15T/long | 0.8667 | 0.8667 | -0.00% | 0.0897 | 0.0897 | -0.00% |
| ett2/15T/medium | 0.8554 | 0.8554 | -0.00% | 0.0875 | 0.0875 | +0.00% |
| ett2/15T/short | 0.7091 | 0.7091 | -0.00% | 0.0599 | 0.0599 | -0.00% |
| ett2/D/short | 1.2566 | 1.2566 | -0.00% | 0.0900 | 0.0900 | +0.00% |
| ett2/H/long | 0.9951 | 0.9947 | +0.04% | 0.0995 | 0.0994 | +0.05% |
| ett2/H/medium | 1.0218 | 1.0218 | -0.00% | 0.1014 | 0.1014 | -0.00% |
| ett2/H/short | 0.7291 | 0.7291 | -0.00% | 0.0630 | 0.0630 | -0.00% |
| ett2/W/short | 0.7642 | 0.7642 | +0.00% | 0.0840 | 0.0840 | +0.00% |
| hierarchical_sales/D/short | 0.7431 | 0.7431 | +0.00% | 0.5683 | 0.5683 | +0.00% |
| hierarchical_sales/W/short | 0.7131 | 0.7131 | -0.00% | 0.3434 | 0.3434 | -0.00% |
| hospital/M/short | 0.7609 | 0.7609 | -0.00% | 0.0512 | 0.0512 | -0.00% |
| jena_weather/10T/long | 0.6064 | 0.6065 | -0.01% | 0.0462 | 0.0462 | -0.09% |
| jena_weather/10T/medium | 0.5595 | 0.5596 | -0.02% | 0.0450 | 0.0450 | -0.19% |
| jena_weather/10T/short | 0.2508 | 0.2509 | -0.03% | 0.0266 | 0.0266 | -0.21% |
| jena_weather/D/short | 1.0231 | 1.0231 | +0.00% | 0.0450 | 0.0450 | +0.00% |
| jena_weather/H/long | 0.8118 | 0.8102 | +0.20% | 0.0517 | 0.0516 | +0.15% |
| jena_weather/H/medium | 0.7584 | 0.7578 | +0.08% | 0.0488 | 0.0488 | -0.01% |
| jena_weather/H/short | 0.5137 | 0.5137 | +0.01% | 0.0405 | 0.0405 | +0.00% |
| kdd_cup_2018/D/short | 1.1937 | 1.1937 | +0.00% | 0.3771 | 0.3771 | -0.00% |
| kdd_cup_2018/H/long | 0.9918 | 0.9918 | +0.00% | 0.4329 | 0.4329 | -0.00% |
| kdd_cup_2018/H/medium | 1.0432 | 1.0432 | -0.00% | 0.4277 | 0.4277 | -0.00% |
| kdd_cup_2018/H/short | 0.9259 | 0.9260 | -0.01% | 0.3693 | 0.3693 | -0.01% |
| loop_seattle/5T/long | 0.7504 | 0.7504 | -0.00% | 0.0664 | 0.0664 | -0.00% |
| loop_seattle/5T/medium | 0.7036 | 0.7036 | +0.00% | 0.0618 | 0.0618 | +0.00% |
| loop_seattle/5T/short | 0.5484 | 0.5484 | +0.00% | 0.0469 | 0.0469 | +0.00% |
| loop_seattle/D/short | 0.8577 | 0.8577 | -0.00% | 0.0412 | 0.0412 | +0.00% |
| loop_seattle/H/long | 0.7325 | 0.7325 | -0.00% | 0.0498 | 0.0498 | -0.00% |
| loop_seattle/H/medium | 0.6802 | 0.6802 | +0.00% | 0.0461 | 0.0461 | +0.00% |
| loop_seattle/H/short | 0.6424 | 0.6425 | -0.01% | 0.0441 | 0.0441 | -0.01% |
| m4_daily/D/short | 3.2504 | 3.2500 | +0.01% | 0.0211 | 0.0211 | +0.01% |
| m4_hourly/H/short | 0.6937 | 0.6937 | +0.00% | 0.0212 | 0.0212 | +0.00% |
| m4_monthly/M/short | 0.9294 | 0.9294 | -0.00% | 0.0925 | 0.0925 | -0.00% |
| m4_quarterly/Q/short | 1.1567 | 1.1567 | -0.00% | 0.0742 | 0.0742 | -0.00% |
| m4_weekly/W/short | 1.8612 | 1.8612 | +0.00% | 0.0355 | 0.0355 | -0.00% |
| m4_yearly/A/short | 3.1463 | 3.1463 | -0.00% | 0.1137 | 0.1137 | -0.00% |
| m_dense/D/short | 0.6170 | 0.6199 | -0.48% | 0.0583 | 0.0586 | -0.42% |
| m_dense/H/long | 0.5857 | 0.5719 | +2.41% | 0.0941 | 0.0912 | +3.25% |
| m_dense/H/medium | 0.6031 | 0.6011 | +0.33% | 0.0986 | 0.0978 | +0.90% |
| m_dense/H/short | 0.7161 | 0.6908 | +3.67% | 0.1152 | 0.1103 | +4.40% |
| restaurant/D/short | 0.6777 | 0.6777 | +0.00% | 0.2542 | 0.2542 | +0.00% |
| saugeen/D/short | 2.7331 | 2.7331 | -0.00% | 0.3211 | 0.3211 | -0.00% |
| saugeen/M/short | 0.7097 | 0.7097 | +0.00% | 0.2845 | 0.2845 | -0.00% |
| saugeen/W/short | 1.1762 | 1.1762 | +0.00% | 0.3500 | 0.3500 | +0.00% |
| solar/10T/long | 0.8114 | 0.8114 | +0.00% | 0.3100 | 0.3100 | -0.00% |
| solar/10T/medium | 0.8149 | 0.8149 | +0.00% | 0.3194 | 0.3194 | +0.00% |
| solar/10T/short | 0.8940 | 0.8940 | +0.00% | 0.4392 | 0.4392 | +0.00% |
| solar/D/short | 0.9756 | 0.9756 | -0.00% | 0.2720 | 0.2720 | -0.00% |
| solar/H/long | 0.9122 | 0.9122 | +0.00% | 0.3222 | 0.3222 | -0.00% |
| solar/H/medium | 0.8657 | 0.8657 | -0.00% | 0.3245 | 0.3245 | -0.00% |
| solar/H/short | 0.8655 | 0.8655 | +0.00% | 0.3202 | 0.3201 | +0.02% |
| solar/W/short | 1.1115 | 1.1115 | +0.00% | 0.1495 | 0.1495 | +0.00% |
| sz_taxi/15T/long | 0.5027 | 0.5027 | -0.00% | 0.1953 | 0.1953 | +0.00% |
| sz_taxi/15T/medium | 0.5336 | 0.5336 | +0.00% | 0.2007 | 0.2007 | +0.00% |
| sz_taxi/15T/short | 0.5411 | 0.5411 | +0.00% | 0.1992 | 0.1992 | -0.00% |
| sz_taxi/H/short | 0.5562 | 0.5562 | +0.00% | 0.1338 | 0.1338 | -0.00% |
| temperature_rain/D/short | 1.3356 | 1.3356 | +0.00% | 0.5467 | 0.5467 | -0.00% |
| us_births/D/short | 0.3137 | 0.3137 | +0.00% | 0.0162 | 0.0162 | -0.00% |
| us_births/M/short | 0.5556 | 0.5556 | +0.00% | 0.0113 | 0.0113 | +0.00% |
| us_births/W/short | 0.9954 | 0.9954 | -0.00% | 0.0116 | 0.0116 | +0.00% |

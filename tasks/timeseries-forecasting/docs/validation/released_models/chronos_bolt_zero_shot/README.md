# Chronos-Bolt base zero-shot (`amazon/chronos-bolt-base`)

Zero-shot replay of the GIFT-Eval leaderboard row `chronos_bolt_base` under the
task contract `timeseries-forecast/v1`.

    bash infer.sh INPUT_DIR OUTPUT_DIR

Reads `INPUT_DIR/index.json` + `INPUT_DIR/cases/<config_id>.h5`, writes
`OUTPUT_DIR/cases/<config_id>.h5` (mean `(N, D, H)`, quantiles `(N, D, 9, H)`
for the deciles 0.1 .. 0.9) and `OUTPUT_DIR/model_manifest.json`.

## What it replays

The official GIFT-Eval notebook [`notebooks/chronos.ipynb`](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos.ipynb),
run through the leaderboard's own protocol (rolling windows, per-config MASE
and CRPS/WQL). Model settings are taken from that notebook verbatim.

## Pins

| what | value |
|---|---|
| checkpoint | `amazon/chronos-bolt-base` |
| revision | `5d9f166d69f47aef3401367a7b842e78fe97b121` |
| license | Apache-2.0 |
| parameters | 205,292,928 |
| library | `chronos-forecasting==2.3.2` |
| torch | `2.14.0+cu130` |
| transformers | `5.17.0` |
| python | 3.11 |

The checkpoint is **not** vendored: `predict.py` calls
`huggingface_hub.snapshot_download(repo_id, revision=...)` at run time (cache
directory overridable with `$CHRONOS2_DIR` / `$CHRONOS_BOLT_DIR`).

## Settings

```python
pipe = BaseChronosPipeline.from_pretrained(<chronos-bolt-base>, device_map="cuda:0")
q, m = pipe.predict_quantiles(
    [torch.tensor(ctx_v) for ...],          # one 1-D series per (instance, variate)
    prediction_length=H,
    quantile_levels=[0.1, 0.2, ..., 0.9],
)
```

| setting | value | source |
|---|---|---|
| pipeline | `chronos.ChronosBoltPipeline` | notebook `BaseChronosPipeline.from_pretrained` |
| device / dtype | `device_map="cuda:0"`, checkpoint dtype float32 | notebook + leaderboard `config.json` |
| quantiles | 0.1 ... 0.9 (the model's native nine) | notebook builds `QuantileForecast` from `pipeline.quantiles` |
| point forecast | 0.5 quantile | gluonts `QuantileForecast.mean` falls back to the median |
| `num_samples` | unused | `forecast_type == QUANTILES`, so the notebook passes no `num_samples` |
| context | model cap 2048 (pipeline truncates) | `chronos_config.context_length` |
| native horizon | 64 | `chronos_config.prediction_length` |
| `limit_prediction_length` | `False` (library default, notebook does not set it) | longer horizons are produced by the pipeline's autoregressive quantile-path heuristic |
| multivariate | `to_univariate`: every variate forecast independently | notebook `to_univariate = target_dim > 1` |
| missing values | NaN passed straight to the model | Chronos handles NaN natively |
| batch size | 256 (default; 28 when H > 64) | notebook uses 1024 -- see note below |

The notebook calls `pipeline.predict(...)` and indexes the model's native
quantiles; `predict_quantiles(..., quantile_levels=[0.1..0.9])` takes the same
branch (`set(quantile_levels).issubset(training_quantile_levels)`) and is
numerically identical, but also returns the median as the point forecast.

**Batch size is numerically inert for Chronos-Bolt**: series never interact
inside a batch (the left NaN padding is masked out), so batching only trades
memory for speed. The default is 256 rather than the notebook's 1024 because
the GPU used here is shared; for `H > 64` it drops to 256 / 9 = 28 because the
long-horizon heuristic expands the batch by the nine quantile paths.


## How to run

```bash
uv venv --python 3.11 .venv-chronos
uv pip install --python .venv-chronos/bin/python torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-chronos/bin/python \
    h5py numpy pandas huggingface_hub chronos-forecasting==2.3.2

PATH=$PWD/.venv-chronos/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
    bash infer.sh /path/to/test/input /path/to/test/output
```

## Runtime

Full GIFT-Eval test split: 97 configs, 326,490 instances, on one shared
NVIDIA A100-SXM4-40GB (a ~28 GB training job and another inference job were
resident on the same GPU throughout, so this is a pessimistic wall clock).

| | |
|---|---|
| wall clock | 12:18:12 (44,292 s), including about 5.6 h paused while another job held the GPU |
| peak GPU memory (this process) | 3.44 GiB |
| OOM-driven batch-size reductions | 0 |

> wall clock includes a ~5.6 h external SIGSTOP pause while the GPU was handed to a higher-priority training job; effective compute is ~6.7 h

## Deviations from the official notebook

1. batch size 256 instead of the notebook's 1024 (28 for H>64): numerically inert for Chronos-Bolt because series do not interact inside a batch; chosen because the GPU is shared
2. chronos-forecasting 2.3.2 instead of the (unpinned) version current when the leaderboard row was produced. From 2.1.0 onwards Chronos-Bolt's long-horizon (H>64) heuristic feeds all nine quantile paths back into the context and takes empirical quantiles of the resulting 81 paths; version <=2.0.0 fed only the median path back (see `central_idx` in chronos/chronos_bolt.py). This is the sole cause of the 22 flagged configs: all are medium/long term, MASE (a median statistic) agrees to 0.25% in aggregate and to 8.5% at worst (two configs above 5%) while WQL improves by 5-10%. Pinning chronos-forecasting<=2.0.0 would reproduce the leaderboard's long-horizon WQL.

## Comparison with the leaderboard

Leaderboard CSV: `GIFT-Eval-space/results/chronos_bolt_base/all_results.csv`.
Columns `eval_metrics/MASE[0.5]` and
`eval_metrics/mean_weighted_sum_quantile_loss`, keyed by
`<dataset>/<freq>/<term>`.

| | MASE | WQL |
|---|---|---|
| max abs. relative deviation | 8.516% | 9.957% |
| median abs. relative deviation | 0.025% | 0.024% |
| configs off by > 5% | 2 | 20 |

Aggregate over the 97 configs, geometric mean of
(model / Seasonal_Naive), the leaderboard's own aggregation:

| | MASE | WQL |
|---|---|---|
| this package | **0.8096** | **0.5622** |
| leaderboard CSV recomputed the same way | 0.8076 | 0.5743 |
| published leaderboard figure | 0.8076 | 0.5743 |

Per config:

| config | MASE ours | MASE leaderboard | dev | WQL ours | WQL leaderboard | dev |
|---|---|---|---|---|---|---|
| bitbrains_fast_storage/5T/long | 0.9535 | 0.9483 | +0.55% | 0.7531 | 0.7475 | +0.74% |
| bitbrains_fast_storage/5T/medium | 1.0480 | 1.0624 | -1.36% | 0.6931 | 0.7554 | -8.25% |
| bitbrains_fast_storage/5T/short | 0.7448 | 0.7524 | -1.00% | 0.4532 | 0.4535 | -0.07% |
| bitbrains_fast_storage/H/short | 1.0697 | 1.0699 | -0.02% | 0.7736 | 0.7740 | -0.05% |
| bitbrains_rnd/5T/long | 3.4083 | 3.3974 | +0.32% | 0.6877 | 0.7561 | -9.05% |
| bitbrains_rnd/5T/medium | 4.4599 | 4.4491 | +0.24% | 0.6030 | 0.6049 | -0.31% |
| bitbrains_rnd/5T/short | 1.7053 | 1.7054 | -0.01% | 0.4381 | 0.4383 | -0.05% |
| bitbrains_rnd/H/short | 5.8969 | 5.8969 | -0.00% | 0.6236 | 0.6242 | -0.10% |
| bizitobs_application/10S/long | 10.2474 | 10.4838 | -2.26% | 0.1050 | 0.1090 | -3.62% |
| bizitobs_application/10S/medium | 9.5763 | 9.7203 | -1.48% | 0.0995 | 0.1035 | -3.84% |
| bizitobs_application/10S/short | 5.5333 | 5.5333 | -0.00% | 0.0540 | 0.0540 | +0.00% |
| bizitobs_l2c/5T/long | 1.2433 | 1.2407 | +0.21% | 0.7141 | 0.7380 | -3.24% |
| bizitobs_l2c/5T/medium | 0.8759 | 0.8777 | -0.21% | 0.4157 | 0.4454 | -6.68% |
| bizitobs_l2c/5T/short | 0.2782 | 0.2782 | -0.00% | 0.0744 | 0.0744 | -0.00% |
| bizitobs_l2c/H/long | 0.5652 | 0.5557 | +1.72% | 0.2684 | 0.2776 | -3.34% |
| bizitobs_l2c/H/medium | 0.5044 | 0.4952 | +1.86% | 0.2463 | 0.2544 | -3.21% |
| bizitobs_l2c/H/short | 0.4323 | 0.4323 | +0.00% | 0.1890 | 0.1890 | -0.00% |
| bizitobs_service/10S/long | 5.1903 | 5.2975 | -2.02% | 0.1091 | 0.1128 | -3.32% |
| bizitobs_service/10S/medium | 4.9254 | 4.9766 | -1.03% | 0.0938 | 0.0962 | -2.42% |
| bizitobs_service/10S/short | 3.2986 | 3.3162 | -0.53% | 0.0500 | 0.0507 | -1.28% |
| car_parts/M/short | 0.8549 | 0.8551 | -0.02% | 0.9944 | 0.9945 | -0.01% |
| covid_deaths/D/short | 38.8419 | 38.8548 | -0.03% | 0.0470 | 0.0470 | -0.00% |
| electricity/15T/long | 0.9461 | 0.9325 | +1.45% | 0.0812 | 0.0842 | -3.57% |
| electricity/15T/medium | 0.8661 | 0.8616 | +0.52% | 0.0800 | 0.0827 | -3.35% |
| electricity/15T/short | 0.9355 | 0.9355 | -0.00% | 0.0817 | 0.0817 | -0.00% |
| electricity/D/short | 1.4478 | 1.4478 | +0.00% | 0.0553 | 0.0553 | -0.00% |
| electricity/H/long | 1.2424 | 1.2375 | +0.40% | 0.0920 | 0.0978 | -5.91% |
| electricity/H/medium | 1.0819 | 1.0781 | +0.35% | 0.0776 | 0.0810 | -4.21% |
| electricity/H/short | 0.8733 | 0.8733 | -0.00% | 0.0636 | 0.0636 | -0.00% |
| electricity/W/short | 1.4765 | 1.4766 | -0.01% | 0.0473 | 0.0473 | -0.00% |
| ett1/15T/long | 1.1259 | 1.1358 | -0.87% | 0.2803 | 0.2978 | -5.87% |
| ett1/15T/medium | 1.0544 | 1.0613 | -0.65% | 0.2617 | 0.2809 | -6.86% |
| ett1/15T/short | 0.6797 | 0.6797 | +0.00% | 0.1579 | 0.1579 | -0.00% |
| ett1/D/short | 1.6723 | 1.6723 | +0.00% | 0.2873 | 0.2873 | +0.00% |
| ett1/H/long | 1.3456 | 1.3544 | -0.65% | 0.2868 | 0.3109 | -7.76% |
| ett1/H/medium | 1.3615 | 1.3748 | -0.96% | 0.2786 | 0.3027 | -7.97% |
| ett1/H/short | 0.8279 | 0.8279 | -0.00% | 0.1811 | 0.1811 | -0.00% |
| ett1/W/short | 1.6970 | 1.6970 | +0.00% | 0.2958 | 0.2958 | +0.00% |
| ett2/15T/long | 0.9456 | 0.9400 | +0.60% | 0.1051 | 0.1114 | -5.73% |
| ett2/15T/medium | 0.9244 | 0.9223 | +0.22% | 0.1022 | 0.1104 | -7.43% |
| ett2/15T/short | 0.7660 | 0.7661 | -0.00% | 0.0666 | 0.0666 | -0.00% |
| ett2/D/short | 1.3216 | 1.3216 | +0.00% | 0.0935 | 0.0935 | +0.00% |
| ett2/H/long | 1.0719 | 1.0364 | +3.42% | 0.1136 | 0.1169 | -2.79% |
| ett2/H/medium | 1.0501 | 1.0273 | +2.22% | 0.1098 | 0.1147 | -4.23% |
| ett2/H/short | 0.7329 | 0.7329 | +0.00% | 0.0634 | 0.0634 | -0.00% |
| ett2/W/short | 0.7391 | 0.7391 | +0.00% | 0.0882 | 0.0882 | -0.00% |
| hierarchical_sales/D/short | 0.7428 | 0.7428 | -0.00% | 0.5761 | 0.5761 | -0.00% |
| hierarchical_sales/W/short | 0.7327 | 0.7327 | -0.01% | 0.3526 | 0.3526 | -0.01% |
| hospital/M/short | 0.7913 | 0.7913 | +0.00% | 0.0571 | 0.0571 | -0.00% |
| jena_weather/10T/long | 0.6536 | 0.6567 | -0.48% | 0.0587 | 0.0638 | -8.12% |
| jena_weather/10T/medium | 0.6078 | 0.6102 | -0.39% | 0.0529 | 0.0571 | -7.46% |
| jena_weather/10T/short | 0.3054 | 0.3055 | -0.02% | 0.0326 | 0.0326 | -0.05% |
| jena_weather/D/short | 1.0511 | 1.0511 | +0.00% | 0.0450 | 0.0450 | -0.00% |
| jena_weather/H/long | 1.0180 | 1.0281 | -0.98% | 0.0584 | 0.0616 | -5.18% |
| jena_weather/H/medium | 0.7466 | 0.7473 | -0.09% | 0.0502 | 0.0540 | -6.95% |
| jena_weather/H/short | 0.5358 | 0.5358 | +0.00% | 0.0422 | 0.0422 | -0.01% |
| kdd_cup_2018/D/short | 1.1967 | 1.1967 | -0.00% | 0.3725 | 0.3725 | +0.00% |
| kdd_cup_2018/H/long | 0.6892 | 0.6843 | +0.71% | 0.2930 | 0.3000 | -2.32% |
| kdd_cup_2018/H/medium | 0.7008 | 0.6999 | +0.13% | 0.2932 | 0.3011 | -2.64% |
| kdd_cup_2018/H/short | 0.6007 | 0.6007 | -0.00% | 0.2464 | 0.2464 | -0.00% |
| loop_seattle/5T/long | 1.2073 | 1.2386 | -2.53% | 0.1172 | 0.1290 | -9.13% |
| loop_seattle/5T/medium | 1.1165 | 1.1356 | -1.68% | 0.1066 | 0.1160 | -8.06% |
| loop_seattle/5T/short | 0.6276 | 0.6276 | -0.00% | 0.0546 | 0.0546 | -0.00% |
| loop_seattle/D/short | 0.9030 | 0.9030 | -0.00% | 0.0437 | 0.0437 | -0.00% |
| loop_seattle/H/long | 1.0269 | 0.9956 | +3.14% | 0.0735 | 0.0759 | -3.17% |
| loop_seattle/H/medium | 1.0542 | 1.0187 | +3.49% | 0.0750 | 0.0762 | -1.53% |
| loop_seattle/H/short | 0.8996 | 0.8996 | +0.00% | 0.0650 | 0.0650 | -0.00% |
| m4_daily/D/short | 3.1985 | 3.1985 | -0.00% | 0.0211 | 0.0211 | -0.00% |
| m4_hourly/H/short | 0.8365 | 0.8365 | +0.00% | 0.0253 | 0.0253 | -0.00% |
| m4_monthly/M/short | 0.9488 | 0.9488 | +0.00% | 0.0938 | 0.0938 | -0.00% |
| m4_quarterly/Q/short | 1.2239 | 1.2239 | +0.00% | 0.0771 | 0.0771 | -0.00% |
| m4_weekly/W/short | 2.0776 | 2.0776 | -0.00% | 0.0382 | 0.0382 | -0.00% |
| m4_yearly/A/short | 3.5066 | 3.5066 | +0.00% | 0.1215 | 0.1215 | -0.00% |
| m_dense/D/short | 0.7163 | 0.7163 | +0.00% | 0.0690 | 0.0690 | +0.00% |
| m_dense/H/long | 0.9935 | 0.9382 | +5.89% | 0.1707 | 0.1698 | +0.51% |
| m_dense/H/medium | 0.9062 | 0.8805 | +2.91% | 0.1554 | 0.1573 | -1.19% |
| m_dense/H/short | 0.7754 | 0.7754 | -0.00% | 0.1250 | 0.1250 | -0.00% |
| restaurant/D/short | 0.7004 | 0.7004 | -0.00% | 0.2640 | 0.2640 | -0.00% |
| saugeen/D/short | 2.8399 | 2.8399 | -0.00% | 0.3380 | 0.3380 | -0.00% |
| saugeen/M/short | 0.7389 | 0.7389 | +0.00% | 0.2962 | 0.2962 | -0.00% |
| saugeen/W/short | 1.2161 | 1.2161 | +0.00% | 0.3627 | 0.3627 | +0.00% |
| solar/10T/long | 1.0918 | 1.0707 | +1.97% | 0.4093 | 0.4430 | -7.61% |
| solar/10T/medium | 1.0111 | 1.0268 | -1.52% | 0.3927 | 0.4361 | -9.96% |
| solar/10T/short | 0.9900 | 0.9907 | -0.06% | 0.5112 | 0.5113 | -0.02% |
| solar/D/short | 0.9818 | 0.9818 | +0.00% | 0.2874 | 0.2874 | +0.00% |
| solar/H/long | 1.1210 | 1.0330 | +8.52% | 0.4190 | 0.4052 | +3.42% |
| solar/H/medium | 0.9686 | 0.9312 | +4.02% | 0.3685 | 0.3685 | +0.00% |
| solar/H/short | 0.8129 | 0.8127 | +0.03% | 0.2977 | 0.2979 | -0.07% |
| solar/W/short | 0.9797 | 0.9797 | +0.00% | 0.1326 | 0.1326 | -0.00% |
| sz_taxi/15T/long | 0.5478 | 0.5455 | +0.43% | 0.2288 | 0.2485 | -7.92% |
| sz_taxi/15T/medium | 0.5618 | 0.5594 | +0.44% | 0.2222 | 0.2444 | -9.06% |
| sz_taxi/15T/short | 0.5481 | 0.5481 | +0.00% | 0.2019 | 0.2019 | -0.00% |
| sz_taxi/H/short | 0.5624 | 0.5624 | -0.00% | 0.1359 | 0.1359 | -0.00% |
| temperature_rain/D/short | 1.3041 | 1.3042 | -0.01% | 0.5377 | 0.5377 | -0.01% |
| us_births/D/short | 0.4847 | 0.4847 | -0.00% | 0.0259 | 0.0259 | -0.00% |
| us_births/M/short | 0.9240 | 0.9240 | +0.00% | 0.0195 | 0.0195 | +0.00% |
| us_births/W/short | 1.0875 | 1.0875 | +0.00% | 0.0129 | 0.0129 | +0.00% |

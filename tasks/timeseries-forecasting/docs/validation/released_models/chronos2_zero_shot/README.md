# Chronos-2 zero-shot (`amazon/chronos-2`)

Zero-shot replay of the GIFT-Eval leaderboard row `Chronos-2` under the
task contract `timeseries-forecast/v1`.

    bash infer.sh INPUT_DIR OUTPUT_DIR

Reads `INPUT_DIR/index.json` + `INPUT_DIR/cases/<config_id>.h5`, writes
`OUTPUT_DIR/cases/<config_id>.h5` (mean `(N, D, H)`, quantiles `(N, D, 9, H)`
for the deciles 0.1 .. 0.9) and `OUTPUT_DIR/model_manifest.json`.

## What it replays

The official GIFT-Eval notebook [`notebooks/chronos-2.ipynb`](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos-2.ipynb),
run through the leaderboard's own protocol (rolling windows, per-config MASE
and CRPS/WQL). Model settings are taken from that notebook verbatim.

## Pins

| what | value |
|---|---|
| checkpoint | `amazon/chronos-2` |
| revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| license | Apache-2.0 |
| parameters | 119,477,664 |
| library | `chronos-forecasting==2.3.2` |
| torch | `2.14.0+cu130` |
| transformers | `5.17.0` |
| python | 3.11 |

The checkpoint is **not** vendored: `predict.py` calls
`huggingface_hub.snapshot_download(repo_id, revision=...)` at run time (cache
directory overridable with `$CHRONOS2_DIR` / `$CHRONOS_BOLT_DIR`).

## Settings

```python
pipe = BaseChronosPipeline.from_pretrained(<chronos-2>, device_map="cuda", torch_dtype="float32")
# one rolling window at a time, items in dataset order
q, m = pipe.predict_quantiles(
    inputs=[{"target": ctx}, ...],          # ctx is (T,) univariate or (D, T) multivariate
    prediction_length=H,
    batch_size=100,
    quantile_levels=[0.1, 0.2, ..., 0.9],
    cross_learning=True,                    # notebook: predict_batches_jointly=True
)
```

| setting | value | source |
|---|---|---|
| pipeline | `chronos.Chronos2Pipeline` | notebook asserts the type |
| device / dtype | `device_map="cuda"`, `torch_dtype="float32"` | notebook `evaluate_on_dataset(...)` |
| batch size | 100 | notebook `batch_size=100` |
| cross-learning | on | notebook `predict_batches_jointly=True` |
| rolling windows | predicted one window at a time | notebook `itertools.islice(test_data.input, window_idx, None, n_windows)` |
| multivariate | native `(D, T)` target | notebook `use_multivariate_data=True` |
| quantiles | 0.1 ... 0.9 | notebook `quantile_levels` |
| point forecast | 0.5 quantile | gluonts `QuantileForecast.mean` falls back to the median |
| context | model cap 8192 (pipeline truncates) | `chronos_config.context_length` |
| horizon | direct, up to 1024 steps | `max_output_patches` 64 x `output_patch_size` 16 |
| missing values | NaN passed straight to the model | Chronos-2 handles NaN natively |

`predict_batches_jointly` is the deprecated spelling of `cross_learning` in
chronos-forecasting >= 2.2 (it emits a `FutureWarning` and assigns to
`cross_learning`); this package passes `cross_learning=True`, the identical
code path.

Cross-learning makes each forecast depend on the other members of its batch, so
batch composition matters. The staged case files are ordered item-major /
window-minor, exactly like gluonts `test_data.input`, so selecting
`window == k` in file order reproduces the notebook's batches item for item.


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
| wall clock | 1:27:07 (5,211 s) |
| peak GPU memory (this process) | 2.58 GiB |
| OOM-driven batch-size reductions | 0 |

## Deviations from the official notebook

1. none: settings taken verbatim from gift-eval notebooks/chronos-2.ipynb; predict_batches_jointly=True passed as its non-deprecated alias cross_learning=True

## Comparison with the leaderboard

Leaderboard CSV: `GIFT-Eval-space/results/chronos-2/all_results.csv`.
Columns `eval_metrics/MASE[0.5]` and
`eval_metrics/mean_weighted_sum_quantile_loss`, keyed by
`<dataset>/<freq>/<term>`.

| | MASE | WQL |
|---|---|---|
| max abs. relative deviation | 0.039% | 0.014% |
| median abs. relative deviation | 0.000% | 0.000% |
| configs off by > 5% | 0 | 0 |

Aggregate over the 97 configs, geometric mean of
(model / Seasonal_Naive), the leaderboard's own aggregation:

| | MASE | WQL |
|---|---|---|
| this package | **0.6978** | **0.4854** |
| leaderboard CSV recomputed the same way | 0.6978 | 0.4854 |
| published leaderboard figure | 0.6978 | 0.4854 |

Per config:

| config | MASE ours | MASE leaderboard | dev | WQL ours | WQL leaderboard | dev |
|---|---|---|---|---|---|---|
| bitbrains_fast_storage/5T/long | 0.8711 | 0.8712 | -0.01% | 0.7025 | 0.7026 | -0.01% |
| bitbrains_fast_storage/5T/medium | 0.9587 | 0.9588 | -0.01% | 0.6233 | 0.6233 | -0.01% |
| bitbrains_fast_storage/5T/short | 0.6563 | 0.6564 | -0.02% | 0.3906 | 0.3906 | -0.01% |
| bitbrains_fast_storage/H/short | 0.9309 | 0.9310 | -0.01% | 0.6635 | 0.6635 | -0.01% |
| bitbrains_rnd/5T/long | 3.2753 | 3.2754 | -0.00% | 0.8729 | 0.8730 | -0.01% |
| bitbrains_rnd/5T/medium | 4.3592 | 4.3592 | -0.00% | 1.0045 | 1.0046 | -0.01% |
| bitbrains_rnd/5T/short | 1.6181 | 1.6181 | -0.00% | 0.4157 | 0.4157 | -0.01% |
| bitbrains_rnd/H/short | 5.8047 | 5.8047 | -0.00% | 0.8035 | 0.8035 | -0.01% |
| bizitobs_application/10S/long | 2.9497 | 2.9497 | -0.00% | 0.0454 | 0.0454 | -0.00% |
| bizitobs_application/10S/medium | 1.8026 | 1.8026 | -0.00% | 0.0257 | 0.0257 | -0.00% |
| bizitobs_application/10S/short | 1.0039 | 1.0039 | -0.00% | 0.0096 | 0.0096 | -0.00% |
| bizitobs_l2c/5T/long | 0.6471 | 0.6472 | -0.01% | 0.2978 | 0.2978 | -0.00% |
| bizitobs_l2c/5T/medium | 0.5826 | 0.5826 | -0.00% | 0.2472 | 0.2472 | -0.00% |
| bizitobs_l2c/5T/short | 0.2665 | 0.2665 | -0.00% | 0.0691 | 0.0691 | -0.00% |
| bizitobs_l2c/H/long | 0.5712 | 0.5713 | -0.01% | 0.2673 | 0.2673 | -0.00% |
| bizitobs_l2c/H/medium | 0.4979 | 0.4980 | -0.01% | 0.2359 | 0.2359 | -0.00% |
| bizitobs_l2c/H/short | 0.4126 | 0.4126 | -0.01% | 0.1765 | 0.1765 | -0.00% |
| bizitobs_service/10S/long | 1.3256 | 1.3256 | +0.00% | 0.0511 | 0.0511 | -0.00% |
| bizitobs_service/10S/medium | 0.9791 | 0.9790 | +0.00% | 0.0221 | 0.0221 | -0.00% |
| bizitobs_service/10S/short | 0.7170 | 0.7169 | +0.01% | 0.0101 | 0.0101 | -0.00% |
| car_parts/M/short | 0.8359 | 0.8360 | -0.00% | 0.9660 | 0.9660 | -0.00% |
| covid_deaths/D/short | 32.5389 | 32.5392 | -0.00% | 0.0348 | 0.0348 | -0.00% |
| electricity/15T/long | 0.8554 | 0.8554 | -0.00% | 0.0717 | 0.0717 | -0.00% |
| electricity/15T/medium | 0.8041 | 0.8041 | -0.00% | 0.0714 | 0.0714 | -0.00% |
| electricity/15T/short | 0.9223 | 0.9223 | -0.00% | 0.0784 | 0.0784 | -0.00% |
| electricity/D/short | 1.4148 | 1.4149 | -0.00% | 0.0581 | 0.0581 | -0.00% |
| electricity/H/long | 1.1925 | 1.1925 | -0.00% | 0.0877 | 0.0878 | -0.00% |
| electricity/H/medium | 1.0692 | 1.0692 | -0.00% | 0.0759 | 0.0759 | -0.00% |
| electricity/H/short | 0.9779 | 0.9779 | -0.00% | 0.0679 | 0.0679 | -0.00% |
| electricity/W/short | 1.3851 | 1.3851 | -0.00% | 0.0559 | 0.0559 | -0.00% |
| ett1/15T/long | 1.0134 | 1.0134 | -0.00% | 0.2411 | 0.2411 | -0.00% |
| ett1/15T/medium | 0.9860 | 0.9860 | +0.00% | 0.2330 | 0.2330 | +0.00% |
| ett1/15T/short | 0.6869 | 0.6869 | +0.00% | 0.1653 | 0.1653 | +0.00% |
| ett1/D/short | 1.6403 | 1.6403 | -0.00% | 0.2742 | 0.2742 | -0.00% |
| ett1/H/long | 1.3805 | 1.3805 | -0.00% | 0.2752 | 0.2752 | -0.00% |
| ett1/H/medium | 1.2738 | 1.2738 | -0.00% | 0.2606 | 0.2606 | +0.00% |
| ett1/H/short | 0.7868 | 0.7868 | +0.00% | 0.1706 | 0.1706 | +0.00% |
| ett1/W/short | 1.5901 | 1.5901 | +0.00% | 0.2714 | 0.2714 | +0.00% |
| ett2/15T/long | 0.8903 | 0.8903 | -0.00% | 0.0929 | 0.0929 | -0.00% |
| ett2/15T/medium | 0.8487 | 0.8487 | +0.00% | 0.0874 | 0.0874 | -0.00% |
| ett2/15T/short | 0.7261 | 0.7262 | -0.00% | 0.0622 | 0.0622 | -0.00% |
| ett2/D/short | 1.2908 | 1.2908 | -0.00% | 0.0942 | 0.0942 | -0.00% |
| ett2/H/long | 1.0480 | 1.0480 | -0.00% | 0.1051 | 0.1051 | +0.00% |
| ett2/H/medium | 1.0902 | 1.0902 | -0.00% | 0.1091 | 0.1091 | -0.00% |
| ett2/H/short | 0.7363 | 0.7363 | -0.00% | 0.0642 | 0.0642 | -0.00% |
| ett2/W/short | 0.7716 | 0.7716 | -0.00% | 0.0896 | 0.0896 | -0.00% |
| hierarchical_sales/D/short | 0.7440 | 0.7440 | +0.00% | 0.5789 | 0.5789 | -0.00% |
| hierarchical_sales/W/short | 0.7093 | 0.7093 | -0.00% | 0.3407 | 0.3407 | -0.00% |
| hospital/M/short | 0.7399 | 0.7399 | +0.00% | 0.0510 | 0.0510 | -0.01% |
| jena_weather/10T/long | 0.6918 | 0.6919 | -0.00% | 0.0514 | 0.0514 | -0.00% |
| jena_weather/10T/medium | 0.6057 | 0.6057 | -0.00% | 0.0496 | 0.0496 | -0.00% |
| jena_weather/10T/short | 0.2649 | 0.2650 | -0.00% | 0.0303 | 0.0303 | -0.01% |
| jena_weather/D/short | 1.0823 | 1.0823 | -0.00% | 0.0467 | 0.0467 | -0.00% |
| jena_weather/H/long | 1.0084 | 1.0084 | -0.00% | 0.0586 | 0.0586 | -0.01% |
| jena_weather/H/medium | 0.7856 | 0.7857 | -0.00% | 0.0503 | 0.0503 | -0.01% |
| jena_weather/H/short | 0.5226 | 0.5226 | -0.00% | 0.0419 | 0.0419 | -0.01% |
| kdd_cup_2018/D/short | 1.1978 | 1.1978 | -0.00% | 0.3666 | 0.3666 | +0.00% |
| kdd_cup_2018/H/long | 1.0469 | 1.0469 | -0.00% | 0.4453 | 0.4453 | -0.00% |
| kdd_cup_2018/H/medium | 1.0401 | 1.0401 | -0.00% | 0.4172 | 0.4172 | -0.00% |
| kdd_cup_2018/H/short | 0.9467 | 0.9467 | -0.00% | 0.3736 | 0.3736 | -0.00% |
| loop_seattle/5T/long | 0.9113 | 0.9113 | +0.00% | 0.0795 | 0.0795 | -0.00% |
| loop_seattle/5T/medium | 0.8455 | 0.8455 | +0.00% | 0.0742 | 0.0742 | -0.00% |
| loop_seattle/5T/short | 0.5407 | 0.5407 | +0.00% | 0.0462 | 0.0462 | -0.00% |
| loop_seattle/D/short | 0.8878 | 0.8878 | +0.00% | 0.0424 | 0.0424 | -0.00% |
| loop_seattle/H/long | 0.8735 | 0.8735 | -0.00% | 0.0593 | 0.0593 | -0.00% |
| loop_seattle/H/medium | 0.9076 | 0.9076 | +0.00% | 0.0628 | 0.0628 | +0.00% |
| loop_seattle/H/short | 0.8250 | 0.8250 | +0.00% | 0.0577 | 0.0577 | -0.00% |
| m4_daily/D/short | 3.4192 | 3.4193 | -0.00% | 0.0229 | 0.0229 | -0.00% |
| m4_hourly/H/short | 0.7884 | 0.7884 | -0.00% | 0.0232 | 0.0232 | -0.00% |
| m4_monthly/M/short | 0.9032 | 0.9032 | -0.00% | 0.0899 | 0.0899 | -0.00% |
| m4_quarterly/Q/short | 1.1542 | 1.1543 | -0.00% | 0.0732 | 0.0732 | -0.00% |
| m4_weekly/W/short | 2.0265 | 2.0263 | +0.01% | 0.0367 | 0.0367 | -0.00% |
| m4_yearly/A/short | 3.2361 | 3.2362 | -0.00% | 0.1115 | 0.1115 | -0.00% |
| m_dense/D/short | 0.6151 | 0.6151 | -0.00% | 0.0578 | 0.0578 | +0.00% |
| m_dense/H/long | 0.6824 | 0.6824 | -0.00% | 0.1133 | 0.1133 | -0.00% |
| m_dense/H/medium | 0.6971 | 0.6971 | -0.00% | 0.1172 | 0.1172 | -0.00% |
| m_dense/H/short | 0.7817 | 0.7817 | +0.00% | 0.1283 | 0.1283 | -0.00% |
| restaurant/D/short | 0.6777 | 0.6777 | +0.00% | 0.2543 | 0.2543 | -0.00% |
| saugeen/D/short | 2.9834 | 2.9834 | -0.00% | 0.3494 | 0.3494 | -0.00% |
| saugeen/M/short | 0.7235 | 0.7235 | +0.00% | 0.2872 | 0.2872 | -0.00% |
| saugeen/W/short | 1.1876 | 1.1876 | -0.00% | 0.3469 | 0.3469 | -0.00% |
| solar/10T/long | 0.7939 | 0.7941 | -0.03% | 0.2900 | 0.2901 | -0.01% |
| solar/10T/medium | 0.7897 | 0.7900 | -0.03% | 0.2902 | 0.2902 | -0.01% |
| solar/10T/short | 0.7782 | 0.7784 | -0.03% | 0.3859 | 0.3860 | -0.01% |
| solar/D/short | 0.9610 | 0.9610 | +0.00% | 0.2666 | 0.2666 | +0.00% |
| solar/H/long | 0.9950 | 0.9951 | -0.01% | 0.3397 | 0.3397 | -0.01% |
| solar/H/medium | 1.0692 | 1.0694 | -0.02% | 0.3629 | 0.3630 | -0.01% |
| solar/H/short | 0.9841 | 0.9845 | -0.04% | 0.3462 | 0.3462 | -0.01% |
| solar/W/short | 1.0123 | 1.0123 | +0.00% | 0.1387 | 0.1387 | +0.00% |
| sz_taxi/15T/long | 0.5097 | 0.5097 | -0.00% | 0.2011 | 0.2011 | -0.00% |
| sz_taxi/15T/medium | 0.5346 | 0.5346 | -0.00% | 0.2020 | 0.2020 | -0.00% |
| sz_taxi/15T/short | 0.5429 | 0.5429 | -0.00% | 0.2001 | 0.2001 | -0.00% |
| sz_taxi/H/short | 0.5603 | 0.5603 | +0.00% | 0.1353 | 0.1353 | -0.00% |
| temperature_rain/D/short | 1.3102 | 1.3102 | -0.00% | 0.5388 | 0.5388 | -0.00% |
| us_births/D/short | 0.3266 | 0.3266 | -0.00% | 0.0165 | 0.0165 | -0.00% |
| us_births/M/short | 0.6328 | 0.6328 | -0.00% | 0.0125 | 0.0125 | -0.00% |
| us_births/W/short | 0.9790 | 0.9790 | -0.00% | 0.0114 | 0.0114 | +0.00% |

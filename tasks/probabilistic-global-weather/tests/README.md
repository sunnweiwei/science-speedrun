# Independent WN2 verifier

Harbor transfers `/workspace/submission` to this separate environment and runs
`bash /tests/test.sh`. The entry point executes the canonical WN2 verifier,
relocates each model and supplies fresh IDs and context-only inputs to an offline
Bubblewrap process. Scoring starts after that process exits. The test truth is
mounted only in the outer trusted scoring environment, not in model inference.

`/opt/weather-python` holds the pinned model runtime; `/opt/wn2-eval` holds the
separate NumPy/xarray/WeatherBench-X CPU evaluator. The default runs all eight cases
at once (one per GPU, eight GPUs) in consecutive windows within a 2 TB allowance, resumes
saved model state, and scores every endpoint with one process per lead. Forecast arrays are deleted only after window scoring and
cross-window precipitation/cyclone capture. Scores, tracks, logs and receipts
remain. Set `WEATHER_EVALUATION_DISK_ALLOWANCE_GB`,
`WEATHER_EVALUATION_DISK_RESERVE_GB`, `WEATHER_EVALUATION_CONCURRENT_CASES` and
`WEATHER_EVALUATION_SCORE_WORKERS` to change planning. See
[windowed evaluation](../wn2/WINDOWED_EVALUATION.md) for accounting, package upgrades
without training, and legacy opt-out requirements.
Invalid/incomplete output fails explicitly;
`/logs/verifier/reward.json` contains both `"score"` and `"reward"` equal to
`1.0 / (1.0 + mean_10_variables(normalized_crps))` (and `0.0` on setup/validation failure),
alongside `normalized_crps` and `normalized_crps_by_variable`. Detailed case/status files
remain under `/logs/verifier/replay/replay-XXX`. A zero after setup failure is not measured
forecast quality. Replay directories must be new.

Both Dockerfiles build from the task root, as specified by the native Compose
files. Set `TASK_PRIVATE_DIR` to the [prepared layout](../environment/data/MATERIALIZE.md).
The verifier Compose configuration permits namespace syscalls in the trusted
outer container; the inner model namespace drops all capabilities. Actual Docker
execution remains unverified. On this HPC host, validated inference uses the
unpacked model image and host Bubblewrap; nested Apptainer checks also require
its root mount to be made private in a fresh user/mount namespace and `/dev/full`.

The canonical isolation implementation is `environment/weathernext/wn2/namespace_inference.py`, copied
into the verifier image under `/tests/wn2/`. The standalone invocation is in
[RUNNING.md](../docs/RUNNING.md).

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary scalar (`score` / `reward` in `/logs/verifier/reward.json`) aggregates latitude-area-weighted **empirical ensemble Continuous Ranked Probability Score** (`CRPS`, `fair=False` in pinned `WeatherBench-X` `CRPSEnsemble`) across the ten native global meteorological fields in `protocol.json` (`geopotential`, `temperature`, `specific_humidity`, `u_component_of_wind`, `v_component_of_wind`, and `vertical_velocity` on all 13 pressure levels `50..1000 hPa`, plus single-level `2m_temperature`, `mean_sea_level_pressure`, `10m_u_component_of_wind`, and `10m_v_component_of_wind`) over 30 twelve-hour leads through 15 days (`12..360 h`) on the native `0.25°` (`721 × 1440`) grid for the 8 seasonally stratified 2022 initialization dates (`56` ensemble members per case):

$$\text{score} = \frac{1}{1 + \frac{1}{10}\sum_{v=1}^{10} \overline{\text{nCRPS}}_v}, \qquad \text{nCRPS}_{v,\ell,\tau} = \frac{\overline{\text{CRPS}}^{\text{emp}}_{v,\ell,\tau}}{\sigma^{\text{train-HRES}}_{v,\ell}}$$

where $\sigma^{\text{train-HRES}}_{v,\ell}$ is the frozen pre-2021 HRES standard deviation (`training-scales.nc`), averaged equally across initialization dates, pressure levels $\ell$, and lead times $\tau$ within each variable $v$, and then equally across the ten variables. This design directly follows:

- **GenCast** (*Price et al., "Probabilistic weather forecasting with machine learning", Nature 2024*, `10.1038/s41586-024-08252-9`) and **WeatherNext 2** (Google DeepMind operational ensemble forecasting protocol), which evaluate medium-range global ensembles through 15 days at `0.25°` against operational analysis (`HRES-fc0`) across atmospheric pressure-level and surface state variables;
- **WeatherBench 2** (*Rasp et al., "WeatherBench 2: A benchmark for the next generation of data-driven global weather models", JAMES 2024*, `10.1029/2023MS004019`), which establishes spherical cell-area-weighted CRPS normalized against climatological/reference dispersion scales as the primary proper scoring rule for probabilistic global NWP.

We explicitly lock **empirical ensemble CRPS (`fair=False`)** rather than finite-sample debiased "fair CRPS" (`fair=True`) because:
1. Both the Agent-facing contract (`instruction.md:10`, `environment/contract.md:111`) and `evaluate.py:77` explicitly specify `empirical ensemble CRPS (fair=False)` on a **fixed, mandatory `M = 56`-member operational product**. Unlike variable-ensemble benchmarks comparing 10-member against 50-member systems, every submission here delivers the exact same `M = 56` ensemble size, eliminating cross-submission ensemble-size bias ($O(1/M)$).
2. Empirical CRPS is a strictly proper scoring rule for the delivered discrete 56-member empirical forecast distribution that downstream decision-makers actually consume, and is guaranteed non-negative panel-by-panel (`CRPS >= 0`), whereas `fair=True` subtracts $\frac{1}{2M(M-1)}\sum_{i \neq j}|x_i - x_j|$ and can yield slightly negative values on small spatial subsets or near-zero-error regimes.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No ad-hoc spread-skill ratio penalty or spectral slope gate in the primary scalar**: Because CRPS is a strictly proper scoring rule over predictive marginals, under-dispersed (overconfident) or over-dispersed ensembles are already directly penalized in CRPS. Adding heuristic spread/skill multipliers or high-wavenumber spectral cutoffs would break properness and incentivize post-hoc smoothing or variance inflation rather than physically calibrated uncertainty.
- **No ad-hoc mixing of ERA5 precipitation or cyclone track error into the HRES state scalar**: Precipitation (`total_precipitation_12hr`) is verified against ERA5 rather than HRES-fc0 and exhibits non-Gaussian dry/light/heavy regime structure best captured by SEEPS, while tropical cyclone tracking depends on discrete TempestExtremes detection counts that vary with storm genesis across dates. Folding heterogeneous truth sources or conditional storm subsets into the primary state scalar with arbitrary weights would obscure physical interpretability.
- **No inference wall-clock shaping inside the 12-hour evaluation budget**: Any self-contained 56-member v3 continuation package completing within the sandbox resource and disk budgets receives its exact physical forecast score.

### 3. Diagnostic Signals Retained in report.json

Although excluded from the single primary scalar, the verifier executes and archives the complete multi-aspect meteorological diagnostic suite under `/logs/verifier/replay/` (`scores/`, `pooling-scores/`, `spectra/`, `cyclone-scores/`, `reward.json`, and `status.json`):
- **Per-variable, per-level, per-lead marginal diagnostics**: `normalized_crps_by_variable`, raw empirical `crps`, ensemble-mean `rmse`, unbiased ensemble `spread`, and `spread_skill` ratio (`sqrt(mean variance / mean MSE)`), from which finite-ensemble fair-CRPS adjustments can also be inspected analytically;
- **Multi-scale spatial & extreme-event diagnostics**: spherical spatial pooling (`pooling-scores/`, evaluating multi-cell neighborhood calibration), Relative Economic Value (`rev_per_case`) across climatological quantile thresholds, and derived wind-speed / thickness panels (`wind_speed`, `10m_wind_speed`, `geopotential_300_minus_500`);
- **Separate ERA5 precipitation & dynamical diagnostics**: 12-hour and 24-hour ERA5 precipitation CRPS/RMSE, Stable Equitable Error in Probability Space (`seeps_per_case`), spherical harmonic zonal/total-wavenumber power spectra (`spectra/`), and 56-member `TempestExtremes` tropical cyclone track/position verification (`cyclone-scores/`).

### Evaluation figures

**Read the complete scorecard for coverage, then the physical-unit curves for error magnitude.** The main model is extended Seed 0 (580 updates); error curves and spectra also retain original Seed 0 (562 updates). All test comparisons use the same eight 2022 dates.

#### Complete evaluation scorecard

![Complete evaluation scorecard](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-scorecard.png?raw=true)

Every one of the 99 field/level combinations and every evaluated lead is shown. The four panels compare extended Seed 0 CRPS and RMSE against GenCast and persistence / dry. Blue means lower error; red means higher error. The color scale saturates at ±200%; exact values remain in the CSV. Blank cells are unavailable. With only eight dates, no statistical-significance annotations are applied.

#### Physical-unit error curves

![Physical-unit error curves](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-errors.png?raw=true)

Temperature, sea-level pressure, Z500, 10 m wind speed and 24 h precipitation: all evaluated leads, with the same eight dates in each model. Lower is better. Curves retain the 12-hour sampling, including the alternating persistence temperature errors; values are not smoothed. These representative curves complement the exhaustive scorecard.

#### Effect of additional autoregressive training

![Extended versus original checkpoint](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-ar-improvement.png?raw=true)

All 99 field/level combinations and evaluated leads compare extended versus original Seed 0 on the full test subset. Positive reductions (blue) mean lower error; colors saturate at ±10%. The CRPS and RMSE panels show where gains concentrate and where errors increase. These correlated panels are descriptive, not independent significance tests.

#### Spatial pooling

![Spatial pooling](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-pooling.png?raw=true)

Average and maximum pooling at all configured mesh refinements and leads. Each cell averages levelwise CRPS percentage reductions within a variable; it is not a physical-unit average across variables. Blue means extended Seed 0 is better than GenCast; colors saturate at ±200%. Blank cells denote unconfigured variable/scale combinations.

#### Spatial power spectra

![Spatial power spectra](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-spectra.png?raw=true)

All ten configured field/level combinations at days 1 and 15, comparing member-mean spectra with truth. Axes are logarithmic; zero wavenumber is omitted. These curves reveal excess or missing power at particular scales, which a total-power ratio can hide. Spectral agreement is a structural diagnostic, not forecast accuracy.

#### Precipitation, extremes and cyclones

![Precipitation, extremes and cyclones](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/evaluation-events.png?raw=true)

SEEPS, extreme-event REV and cyclone position / probability results are shown separately. SEEPS and position error are lower-is-better; REV is higher-is-better, with climatology at zero. Selected REV thresholds and leads are labeled; all settings remain in the complete tables. Paired cyclone reports end at day 9 as matched storm coverage falls from eight to one; later leads are unavailable, not zero error.

[Download complete per-lead values](https://github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/main-evaluation-by-lead.csv). The complete numeric tables remain directly below, including all pressure levels and supplementary metric settings.


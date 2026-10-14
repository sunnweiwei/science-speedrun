# Weather evaluation results

## Evaluation results

Compare **extended Seed 0 (580 updates, main recipe)**, original Seed 0 (562 updates), GenCast and persistence / dry on **eight fixed 2022 dates, 56 members and a 15-day horizon**. Metrics include native and derived CRPS/RMSE, ensemble spread, precipitation, SEEPS, extreme-event REV, spatial pooling, spectra and cyclone tracks. Persistence / dry is defined for core and spatial-pooling metrics; no simple spectral or cyclone-track reference is defined.

| Independent submission metric | Original (562) | Extended (580) |
|---|---:|---:|
| Normalized CRPS ↓ | 0.479717 | **0.469183** |
| Reward ↑ | 0.675805 | **0.680651** |

Additional AR training reduces normalized CRPS by **2.20%**, with lower normalized mean CRPS for all ten native variables. Reward is `1 / (1 + normalized CRPS)`, using training-data scales and equal variable weights; it is not an accuracy percentage. The extended curriculum is the main starter recipe, with four updates at each AR3–AR8 stage. Recipe selection used these test results, so independent confirmation requires new held-out cases.

| Paired comparison | Lower extended CRPS | Lower extended RMSE |
|---|---:|---:|
| Native / derived vs original | 2,827 / 2,969 | 2,758 / 2,969 |
| Spatial pooling vs original | 14,893 / 17,040 | 14,343 / 17,040 |
| Native / derived vs persistence / dry | 2,479 / 2,969 | 1,742 / 2,969 |
| Spatial pooling vs persistence / dry | 10,633 / 17,040 | 5,285 / 17,040 |
| Native / derived vs GenCast | 0 / 2,969 | 0 / 2,969 |
| Spatial pooling vs GenCast | 12 / 17,040 | 8 / 17,040 |

These are counts of correlated variable/level/lead panels; they are not independent trials. Extended SEEPS is lower than original in 80 of 118 panels. Gains are broad but modest and do not close the gap to GenCast. Long-range temperature, circulation and wind errors can still exceed persistence.

Cyclone results are mixed: in the direct original–extended comparison, one-day position error increases from **438.06 to 440.86 km** (nine matched storm initializations), while three-day error decreases from **1,225.04 to 1,221.87 km** and five-day error from **1,661.27 to 1,653.07 km** (seven matched initializations each). These paired storm subsets differ from the GenCast comparison below; their absolute errors must not be compared across unmatched subsets.

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

<details>
<summary>Complete numeric evaluation tables: every field, pressure level and supplementary metric</summary>

### Main evaluation table — all fields and pressure levels

Extended 580-update Seed 0 (main recipe), original 562-update Seed 0, GenCast and persistence / dry, on the **same eight 2022 dates, 56 members and 15-day horizon**. This table includes **all 82 native field/level combinations**, plus 17 precipitation/derived combinations. Lower CRPS and RMSE are better. The simple reference holds initial fields fixed and predicts zero rain.

Each row summarizes **all evaluated leads** (12–360 h every 12 h; 24 h accumulated rain starts at 24 h). CRPS averages equally over dates and leads; RMSE is the square root of the equally averaged MSE, not the mean of lead RMSEs. Pressure levels remain separate. Unit conversions apply equally to all models. Spread is the square root of mean unbiased ensemble variance (ddof=1), in the same units; deterministic simple-reference spread is zero. Spread is a dispersion diagnostic, not an error score. These are physical-unit errors, not the normalized task reward. The temperature-by-lead table below complements this complete field table.

| Variable | hPa / type | Error unit | Leads | Simple CRPS | Original CRPS | Extended CRPS | GenCast CRPS | Simple RMSE | Original RMSE | Extended RMSE | GenCast RMSE | Original spread | Extended spread | GenCast spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geopotential | 50 | m²/s² | 30 | 682 | 647 | 644 | 261 | 1.24e+03 | 1.44e+03 | 1.43e+03 | 584 | 1.47e+03 | 1.45e+03 | 472 |
| geopotential | 100 | m²/s² | 30 | 712 | 730 | 709 | 193 | 1.14e+03 | 1.42e+03 | 1.39e+03 | 484 | 1.92e+03 | 1.88e+03 | 452 |
| geopotential | 150 | m²/s² | 30 | 810 | 761 | 749 | 218 | 1.24e+03 | 1.51e+03 | 1.48e+03 | 536 | 1.73e+03 | 1.71e+03 | 515 |
| geopotential | 200 | m²/s² | 30 | 879 | 891 | 864 | 244 | 1.37e+03 | 1.77e+03 | 1.71e+03 | 612 | 1.91e+03 | 1.87e+03 | 593 |
| geopotential | 250 | m²/s² | 30 | 907 | 938 | 913 | 259 | 1.44e+03 | 1.85e+03 | 1.8e+03 | 668 | 1.94e+03 | 1.91e+03 | 649 |
| geopotential | 300 | m²/s² | 30 | 874 | 934 | 902 | 254 | 1.41e+03 | 1.81e+03 | 1.75e+03 | 664 | 1.88e+03 | 1.85e+03 | 648 |
| geopotential | 400 | m²/s² | 30 | 744 | 846 | 814 | 219 | 1.22e+03 | 1.64e+03 | 1.58e+03 | 579 | 1.99e+03 | 1.95e+03 | 566 |
| geopotential | 500 | m²/s² | 30 | 630 | 751 | 732 | 186 | 1.04e+03 | 1.43e+03 | 1.4e+03 | 493 | 1.8e+03 | 1.77e+03 | 481 |
| geopotential | 600 | m²/s² | 30 | 548 | 627 | 599 | 163 | 901 | 1.22e+03 | 1.16e+03 | 428 | 1.08e+03 | 1.06e+03 | 417 |
| geopotential | 700 | m²/s² | 30 | 491 | 553 | 538 | 147 | 805 | 1.03e+03 | 1.01e+03 | 384 | 1.53e+03 | 1.5e+03 | 372 |
| geopotential | 850 | m²/s² | 30 | 448 | 484 | 459 | 137 | 732 | 948 | 888 | 353 | 796 | 782 | 339 |
| geopotential | 925 | m²/s² | 30 | 453 | 430 | 423 | 139 | 733 | 835 | 817 | 356 | 1.03e+03 | 1.01e+03 | 340 |
| geopotential | 1000 | m²/s² | 30 | 468 | 451 | 425 | 144 | 752 | 889 | 829 | 366 | 824 | 814 | 350 |
| temperature | 50 | K | 30 | 2.42 | 2.76 | 2.69 | 1.47 | 3.58 | 5.31 | 5.16 | 2.78 | 5.67 | 5.63 | 1.78 |
| temperature | 100 | K | 30 | 2.88 | 3.4 | 3.32 | 1.04 | 3.92 | 6.27 | 6.16 | 2.01 | 5.76 | 5.7 | 1.77 |
| temperature | 150 | K | 30 | 2.76 | 3.43 | 3.3 | 0.96 | 4.03 | 6.39 | 6.16 | 2.05 | 5.02 | 4.95 | 1.88 |
| temperature | 200 | K | 30 | 3.07 | 2.19 | 2.18 | 1.05 | 4.68 | 4.31 | 4.27 | 2.44 | 4.28 | 4.22 | 2.33 |
| temperature | 250 | K | 30 | 2.69 | 2.15 | 2.1 | 0.917 | 3.98 | 4.17 | 4.06 | 2.07 | 4.37 | 4.35 | 1.95 |
| temperature | 300 | K | 30 | 2.51 | 2.1 | 2.1 | 0.826 | 3.58 | 3.92 | 3.93 | 1.79 | 5.66 | 5.62 | 1.66 |
| temperature | 400 | K | 30 | 2.83 | 2.59 | 2.52 | 0.908 | 4.11 | 4.88 | 4.78 | 2.02 | 6.23 | 6.1 | 1.92 |
| temperature | 500 | K | 30 | 2.85 | 2.54 | 2.49 | 0.927 | 4.23 | 4.92 | 4.83 | 2.09 | 6.95 | 6.9 | 2.03 |
| temperature | 600 | K | 30 | 2.82 | 2.49 | 2.42 | 0.922 | 4.18 | 4.81 | 4.7 | 2.07 | 5.14 | 5.11 | 2.03 |
| temperature | 700 | K | 30 | 2.82 | 2.63 | 2.62 | 0.925 | 4.19 | 5.02 | 4.99 | 2.09 | 5.84 | 5.83 | 2.03 |
| temperature | 850 | K | 30 | 3.08 | 2.85 | 2.76 | 1.05 | 4.55 | 5.36 | 5.24 | 2.33 | 5.85 | 5.87 | 2.21 |
| temperature | 925 | K | 30 | 2.81 | 2.69 | 2.63 | 0.937 | 4.32 | 5.4 | 5.32 | 2.17 | 5.58 | 5.6 | 2.06 |
| temperature | 1000 | K | 30 | 2.5 | 2.25 | 2.25 | 0.828 | 3.99 | 4.48 | 4.49 | 1.92 | 5.62 | 5.6 | 1.78 |
| specific_humidity | 50 | g/kg | 30 | 5.67e-05 | 6.45e-05 | 6.46e-05 | 4.1e-05 | 0.000135 | 0.000154 | 0.000154 | 9.76e-05 | 0.000129 | 0.000127 | 7.56e-05 |
| specific_humidity | 100 | g/kg | 30 | 0.000275 | 0.000264 | 0.000261 | 0.00014 | 0.000433 | 0.000506 | 0.000499 | 0.000309 | 0.00057 | 0.00056 | 0.000281 |
| specific_humidity | 150 | g/kg | 30 | 0.00187 | 0.00167 | 0.00164 | 0.00084 | 0.00314 | 0.00326 | 0.0032 | 0.00204 | 0.00353 | 0.00349 | 0.00189 |
| specific_humidity | 200 | g/kg | 30 | 0.0132 | 0.0111 | 0.0111 | 0.00557 | 0.0212 | 0.0208 | 0.021 | 0.0132 | 0.0237 | 0.0234 | 0.0126 |
| specific_humidity | 250 | g/kg | 30 | 0.049 | 0.0357 | 0.0355 | 0.02 | 0.0772 | 0.0709 | 0.0705 | 0.047 | 0.0831 | 0.0822 | 0.0449 |
| specific_humidity | 300 | g/kg | 30 | 0.119 | 0.0834 | 0.0834 | 0.0472 | 0.187 | 0.168 | 0.169 | 0.111 | 0.157 | 0.154 | 0.105 |
| specific_humidity | 400 | g/kg | 30 | 0.379 | 0.271 | 0.269 | 0.149 | 0.589 | 0.55 | 0.546 | 0.344 | 0.583 | 0.575 | 0.321 |
| specific_humidity | 500 | g/kg | 30 | 0.766 | 0.522 | 0.52 | 0.3 | 1.15 | 1.03 | 1.03 | 0.67 | 0.969 | 0.959 | 0.634 |
| specific_humidity | 600 | g/kg | 30 | 1.18 | 0.801 | 0.8 | 0.46 | 1.71 | 1.5 | 1.5 | 0.991 | 1.64 | 1.64 | 0.948 |
| specific_humidity | 700 | g/kg | 30 | 1.59 | 1.06 | 1.06 | 0.625 | 2.24 | 1.95 | 1.95 | 1.31 | 2.31 | 2.3 | 1.24 |
| specific_humidity | 850 | g/kg | 30 | 1.78 | 1.22 | 1.22 | 0.734 | 2.49 | 2.26 | 2.25 | 1.51 | 2.28 | 2.27 | 1.47 |
| specific_humidity | 925 | g/kg | 30 | 1.4 | 1.28 | 1.26 | 0.559 | 2.03 | 2.41 | 2.36 | 1.17 | 3.07 | 3.06 | 1.17 |
| specific_humidity | 1000 | g/kg | 30 | 1.36 | 1.18 | 1.17 | 0.538 | 1.97 | 2.26 | 2.23 | 1.11 | 2.84 | 2.81 | 1.03 |
| u_component_of_wind | 50 | m/s | 30 | 5.37 | 5.47 | 5.36 | 2.51 | 7.52 | 10.3 | 10.1 | 4.71 | 11.1 | 11 | 3.36 |
| u_component_of_wind | 100 | m/s | 30 | 7.35 | 6.06 | 5.98 | 2.5 | 9.58 | 10.9 | 10.8 | 4.78 | 9.68 | 9.61 | 4.37 |
| u_component_of_wind | 150 | m/s | 30 | 9.61 | 7.31 | 7.22 | 3.16 | 12.5 | 13.1 | 13 | 6.2 | 11.7 | 11.5 | 5.74 |
| u_component_of_wind | 200 | m/s | 30 | 11.4 | 9.8 | 9.38 | 3.78 | 15.2 | 18.4 | 17.6 | 7.68 | 15.3 | 15.2 | 7.26 |
| u_component_of_wind | 250 | m/s | 30 | 12.2 | 9.02 | 8.9 | 4.13 | 16.5 | 16.3 | 16 | 8.54 | 19.5 | 19.2 | 8.17 |
| u_component_of_wind | 300 | m/s | 30 | 11.7 | 9.41 | 9.05 | 4.04 | 16 | 17.3 | 16.7 | 8.42 | 13.8 | 13.8 | 8.09 |
| u_component_of_wind | 400 | m/s | 30 | 9.73 | 7.51 | 7.24 | 3.43 | 13.3 | 13.9 | 13.4 | 7.12 | 10 | 9.95 | 6.89 |
| u_component_of_wind | 500 | m/s | 30 | 8.09 | 7.06 | 6.69 | 2.87 | 10.9 | 13.3 | 12.6 | 5.88 | 12.2 | 12.2 | 5.72 |
| u_component_of_wind | 600 | m/s | 30 | 7 | 6.34 | 6.03 | 2.49 | 9.42 | 12.1 | 11.5 | 5.06 | 10.2 | 10 | 4.92 |
| u_component_of_wind | 700 | m/s | 30 | 6.27 | 5.5 | 5.26 | 2.25 | 8.43 | 10.3 | 9.87 | 4.56 | 9.57 | 9.48 | 4.41 |
| u_component_of_wind | 850 | m/s | 30 | 5.47 | 3.82 | 3.76 | 2 | 7.58 | 7.12 | 7 | 4.18 | 6.93 | 6.92 | 4.04 |
| u_component_of_wind | 925 | m/s | 30 | 5.17 | 4.09 | 4.02 | 1.88 | 7.42 | 7.85 | 7.7 | 4.13 | 7.49 | 7.41 | 3.97 |
| u_component_of_wind | 1000 | m/s | 30 | 4.03 | 3.29 | 3.23 | 1.48 | 5.84 | 6.22 | 6.07 | 3.25 | 6.45 | 6.42 | 3.11 |
| v_component_of_wind | 50 | m/s | 30 | 4.58 | 3.64 | 3.62 | 1.75 | 6.87 | 6.9 | 6.87 | 3.56 | 7.82 | 7.76 | 3.27 |
| v_component_of_wind | 100 | m/s | 30 | 6.8 | 4.24 | 4.25 | 2.36 | 9.1 | 7.83 | 7.84 | 4.61 | 7.67 | 7.67 | 4.39 |
| v_component_of_wind | 150 | m/s | 30 | 9.39 | 6.47 | 6.47 | 3.15 | 12.5 | 12.2 | 12.2 | 6.27 | 11.6 | 11.6 | 5.93 |
| v_component_of_wind | 200 | m/s | 30 | 11.5 | 7.83 | 7.76 | 3.88 | 15.8 | 14.8 | 14.6 | 8.05 | 15.5 | 15.5 | 7.68 |
| v_component_of_wind | 250 | m/s | 30 | 12.5 | 8.79 | 8.53 | 4.28 | 17.5 | 16.3 | 15.8 | 9.12 | 15.1 | 15 | 8.73 |
| v_component_of_wind | 300 | m/s | 30 | 12 | 7.8 | 7.75 | 4.18 | 17 | 15 | 14.9 | 8.99 | 15.8 | 15.9 | 8.63 |
| v_component_of_wind | 400 | m/s | 30 | 9.88 | 6.17 | 6.09 | 3.49 | 14.1 | 11.9 | 11.8 | 7.5 | 11.5 | 11.6 | 7.22 |
| v_component_of_wind | 500 | m/s | 30 | 8.08 | 5.76 | 5.56 | 2.87 | 11.5 | 10.8 | 10.4 | 6.13 | 10.8 | 10.8 | 5.91 |
| v_component_of_wind | 600 | m/s | 30 | 6.86 | 4.22 | 4.15 | 2.46 | 9.71 | 7.98 | 7.86 | 5.2 | 8.52 | 8.55 | 5.03 |
| v_component_of_wind | 700 | m/s | 30 | 6.03 | 4.04 | 3.93 | 2.19 | 8.51 | 7.81 | 7.49 | 4.59 | 8.85 | 8.82 | 4.44 |
| v_component_of_wind | 850 | m/s | 30 | 5.31 | 3.57 | 3.48 | 1.95 | 7.66 | 6.69 | 6.53 | 4.18 | 7.16 | 7.13 | 4.04 |
| v_component_of_wind | 925 | m/s | 30 | 5.28 | 3.48 | 3.43 | 1.9 | 7.8 | 6.65 | 6.55 | 4.25 | 6.72 | 6.72 | 4.13 |
| v_component_of_wind | 1000 | m/s | 30 | 4.26 | 4.03 | 3.86 | 1.53 | 6.28 | 7.7 | 7.33 | 3.42 | 8.75 | 8.67 | 3.31 |
| vertical_velocity | 50 | Pa/s | 30 | 0.0179 | 0.0133 | 0.0132 | 0.00897 | 0.0278 | 0.0232 | 0.0231 | 0.0198 | 0.044 | 0.0434 | 0.0199 |
| vertical_velocity | 100 | Pa/s | 30 | 0.0378 | 0.0234 | 0.0232 | 0.0186 | 0.0579 | 0.0446 | 0.0444 | 0.0402 | 0.0605 | 0.06 | 0.0408 |
| vertical_velocity | 150 | Pa/s | 30 | 0.0708 | 0.044 | 0.0437 | 0.0346 | 0.11 | 0.0891 | 0.0885 | 0.076 | 0.102 | 0.101 | 0.0783 |
| vertical_velocity | 200 | Pa/s | 30 | 0.104 | 0.0626 | 0.0624 | 0.05 | 0.166 | 0.124 | 0.124 | 0.113 | 0.165 | 0.164 | 0.114 |
| vertical_velocity | 250 | Pa/s | 30 | 0.14 | 0.0819 | 0.0814 | 0.0662 | 0.22 | 0.165 | 0.165 | 0.148 | 0.193 | 0.191 | 0.147 |
| vertical_velocity | 300 | Pa/s | 30 | 0.173 | 0.0984 | 0.0978 | 0.0811 | 0.273 | 0.203 | 0.202 | 0.183 | 0.227 | 0.225 | 0.179 |
| vertical_velocity | 400 | Pa/s | 30 | 0.215 | 0.12 | 0.12 | 0.101 | 0.353 | 0.259 | 0.259 | 0.236 | 0.279 | 0.276 | 0.225 |
| vertical_velocity | 500 | Pa/s | 30 | 0.229 | 0.127 | 0.126 | 0.107 | 0.388 | 0.281 | 0.28 | 0.258 | 0.271 | 0.268 | 0.245 |
| vertical_velocity | 600 | Pa/s | 30 | 0.236 | 0.14 | 0.139 | 0.111 | 0.403 | 0.306 | 0.304 | 0.269 | 0.323 | 0.32 | 0.259 |
| vertical_velocity | 700 | Pa/s | 30 | 0.246 | 0.137 | 0.137 | 0.116 | 0.416 | 0.304 | 0.303 | 0.277 | 0.297 | 0.294 | 0.268 |
| vertical_velocity | 850 | Pa/s | 30 | 0.248 | 0.156 | 0.154 | 0.117 | 0.393 | 0.321 | 0.317 | 0.259 | 0.328 | 0.325 | 0.251 |
| vertical_velocity | 925 | Pa/s | 30 | 0.203 | 0.12 | 0.12 | 0.0953 | 0.309 | 0.244 | 0.243 | 0.2 | 0.258 | 0.257 | 0.199 |
| vertical_velocity | 1000 | Pa/s | 30 | 0.0691 | 0.0583 | 0.0576 | 0.0302 | 0.152 | 0.142 | 0.141 | 0.0815 | 0.126 | 0.124 | 0.079 |
| 2m_temperature | surface / derived | °C error | 30 | 2.4 | 2.29 | 2.18 | 0.752 | 4.02 | 4.68 | 4.55 | 1.78 | 5.85 | 5.86 | 1.66 |
| mean_sea_level_pressure | surface / derived | hPa | 30 | 5.88 | 5.89 | 5.75 | 1.81 | 9.55 | 10.9 | 10.5 | 4.64 | 16.5 | 16.2 | 4.42 |
| 10m_u_component_of_wind | surface / derived | m/s | 30 | 3.68 | 4.26 | 4.13 | 1.36 | 5.29 | 7.96 | 7.7 | 2.96 | 9.86 | 9.75 | 2.81 |
| 10m_v_component_of_wind | surface / derived | m/s | 30 | 3.88 | 3.8 | 3.74 | 1.42 | 5.63 | 7.25 | 7.09 | 3.09 | 8.67 | 8.66 | 2.97 |
| total_precipitation_12hr | surface / derived | mm / 12 h | 30 | 1.47 | 1.34 | 1.32 | 0.85 | 4.52 | 4.24 | 4.22 | 3.57 | 2.55 | 2.54 | 3.33 |
| wind_speed | 50 | m/s | 30 | 4.84 | 5.19 | 5.11 | 2.18 | 7.02 | 10.1 | 9.96 | 4.21 | 8.53 | 8.45 | 3.19 |
| wind_speed | 100 | m/s | 30 | 6.43 | 4.6 | 4.58 | 2.28 | 8.53 | 8.46 | 8.43 | 4.39 | 7.73 | 7.68 | 4.01 |
| wind_speed | 150 | m/s | 30 | 8.59 | 6.14 | 6.13 | 2.92 | 11.3 | 11.4 | 11.5 | 5.72 | 10.3 | 10.3 | 5.28 |
| wind_speed | 200 | m/s | 30 | 10.2 | 7.1 | 7.03 | 3.5 | 13.8 | 13.3 | 13.1 | 7.1 | 12.5 | 12.4 | 6.7 |
| wind_speed | 250 | m/s | 30 | 10.7 | 7.6 | 7.53 | 3.79 | 14.8 | 14.5 | 14.3 | 7.86 | 16 | 15.8 | 7.49 |
| wind_speed | 300 | m/s | 30 | 10.1 | 7.34 | 7.28 | 3.65 | 14.2 | 13.9 | 13.8 | 7.64 | 11.7 | 11.7 | 7.32 |
| wind_speed | 400 | m/s | 30 | 8.22 | 5.3 | 5.24 | 3.03 | 11.6 | 10.1 | 10 | 6.34 | 8.36 | 8.33 | 6.12 |
| wind_speed | 500 | m/s | 30 | 6.73 | 5.03 | 4.86 | 2.5 | 9.38 | 9.72 | 9.37 | 5.16 | 9.86 | 9.65 | 5 |
| wind_speed | 600 | m/s | 30 | 5.73 | 3.99 | 3.88 | 2.15 | 7.9 | 7.77 | 7.53 | 4.36 | 8.13 | 7.93 | 4.24 |
| wind_speed | 700 | m/s | 30 | 5.04 | 3.97 | 3.81 | 1.92 | 6.87 | 7.75 | 7.41 | 3.85 | 7.77 | 7.59 | 3.72 |
| wind_speed | 850 | m/s | 30 | 4.36 | 2.91 | 2.86 | 1.69 | 6.05 | 5.44 | 5.34 | 3.47 | 5.66 | 5.61 | 3.34 |
| wind_speed | 925 | m/s | 30 | 4.11 | 2.88 | 2.84 | 1.58 | 5.79 | 5.42 | 5.33 | 3.32 | 5.78 | 5.7 | 3.21 |
| wind_speed | 1000 | m/s | 30 | 2.91 | 3.04 | 2.94 | 1.13 | 4.03 | 6.09 | 5.85 | 2.33 | 6.73 | 6.56 | 2.24 |
| 10m_wind_speed | surface / derived | m/s | 30 | 2.59 | 4.26 | 4.15 | 1.02 | 3.55 | 8.85 | 8.62 | 2.06 | 8.19 | 8.08 | 1.98 |
| geopotential_300_minus_500 | surface / derived | m²/s² | 30 | 371 | 460 | 456 | 113 | 546 | 938 | 925 | 261 | 1.08e+03 | 1.08e+03 | 249 |
| total_precipitation_24hr | surface / derived | mm / 24 h | 29 | 2.94 | 2.6 | 2.56 | 1.51 | 7.83 | 7.2 | 7.16 | 5.76 | 3.89 | 3.87 | 5.24 |

[Complete table at every evaluated lead: 2,969 rows, all four models](<https:/github.com/T0-RSI/ai4sci-tasks/blob/main/probabilistic-global-weather/docs/validation/results-summary/main-evaluation-by-lead.csv>). No leads or field/level combinations are omitted from this CSV.

### Spatial pooling — all evaluated variable families

All eight dates, both average and maximum pooling, every configured mesh scale, level and lead. Counts below show panels with lower model error than persistence / dry; correlated panels are descriptive, not independent trials. Full absolute scores remain in the aggregate results file.

| Variable | Panels | Extended lower CRPS | Extended lower RMSE | GenCast lower CRPS | GenCast lower RMSE |
|---|---|---|---|---|---|
| geopotential | 3900 | 2030 | 646 | 3900 | 3900 |
| temperature | 3900 | 2446 | 1234 | 3900 | 3892 |
| specific_humidity | 3900 | 2209 | 1133 | 3835 | 3869 |
| 2m_temperature | 360 | 206 | 125 | 360 | 360 |
| mean_sea_level_pressure | 360 | 196 | 133 | 360 | 360 |
| total_precipitation_12hr | 360 | 360 | 360 | 360 | 360 |
| wind_speed | 3900 | 3094 | 1615 | 3900 | 3900 |
| 10m_wind_speed | 360 | 92 | 39 | 360 | 360 |

### Precipitation SEEPS — all accumulation / threshold settings

Equal-weight mean of the eight-date score across all available leads; dimensionless, lower is better.

| Accumulation (h) | Dry threshold (mm) | Leads | Dry SEEPS | Extended SEEPS | GenCast SEEPS |
|---|---|---|---|---|---|
| 12 | 0.1 | 30 | 0.981 | 0.969 | 0.701 |
| 12 | 0.25 | 30 | 0.979 | 0.969 | 0.692 |
| 24 | 0.1 | 29 | 0.976 | 0.979 | 0.679 |
| 24 | 0.25 | 29 | 0.98 | 0.97 | 0.67 |

### Extreme-event REV — all event definitions

Equal-weight mean across all evaluated leads and the configured cost/loss grid, after each paired eight-date aggregation. Larger is better; climatology has REV zero. This grid-average is a descriptive summary, not a newly optimized operating threshold; full curves remain in the aggregate results file.

| Variable | Quantile | Tail | Leads | Simple REV | Extended REV | GenCast REV |
|---|---|---|---|---|---|---|
| 10m_wind_speed | 0.99 | high | 30 | 0.0117 | 0.0266 | 0.203 |
| 10m_wind_speed | 0.999 | high | 30 | 0.021 | 0.0579 | 0.241 |
| 10m_wind_speed | 0.9999 | high | 30 | 0.0247 | 0.0717 | 0.255 |
| 2m_temperature | 0.0001 | low | 30 | 0.054 | 0.0428 | 0.277 |
| 2m_temperature | 0.001 | low | 30 | 0.0465 | 0.0403 | 0.261 |
| 2m_temperature | 0.01 | low | 30 | 0.0304 | 0.0336 | 0.226 |
| 2m_temperature | 0.99 | high | 30 | 0.024 | 0.0595 | 0.212 |
| 2m_temperature | 0.999 | high | 30 | 0.0287 | 0.0889 | 0.221 |
| 2m_temperature | 0.9999 | high | 30 | 0.0303 | 0.0986 | 0.225 |
| mean_sea_level_pressure | 0.0001 | low | 30 | 0.00819 | 0.0319 | 0.426 |
| mean_sea_level_pressure | 0.001 | low | 30 | 0.00849 | 0.0294 | 0.418 |
| mean_sea_level_pressure | 0.01 | low | 30 | 0.0129 | 0.0273 | 0.388 |

### Spatial spectra — every evaluated field / level

Ratio of summed nonzero-wavenumber member-mean power to truth power, averaging power over available dates/leads before division. Truth = 1. This measures total spatial variability, not prediction accuracy or agreement at each wavelength. Persistence spectra are not defined in this evaluation. Full wavelength-dependent reports are separate from this scalar summary.

| Variable | hPa / type | Leads | Extended / truth | GenCast / truth | Truth |
|---|---|---|---|---|---|
| 10m_u_component_of_wind | surface | 30 | 1.94 | 1.01 | 1.000 |
| 2m_temperature | surface | 30 | 1.07 | 1 | 1.000 |
| geopotential | 500 | 30 | 1.37 | 1.01 | 1.000 |
| geopotential | 850 | 30 | 1.58 | 1.01 | 1.000 |
| mean_sea_level_pressure | surface | 30 | 1.83 | 1.03 | 1.000 |
| specific_humidity | 700 | 30 | 0.767 | 0.995 | 1.000 |
| specific_humidity | 925 | 30 | 1.14 | 1.01 | 1.000 |
| temperature | 300 | 30 | 1.17 | 0.995 | 1.000 |
| temperature | 850 | 30 | 1.14 | 1.02 | 1.000 |
| u_component_of_wind | 850 | 30 | 0.787 | 1.04 | 1.000 |

### Cyclones — all evaluated leads

Position errors use storm initializations matched in both forecasts and truth; counts vary with lead. Paired cyclone reports are available only through day 9; days 9.5–15 have no paired report and are unavailable, not zero error. No simple track reference is defined. Position error is in km, lower is better. REV is averaged across its configured cost/loss grid, larger is better.

| Lead (d) | Matched storms | Extended km | GenCast km | Extended REV | GenCast REV |
|---|---|---|---|---|---|
| 0.5 | 8 | 226 | 33.4 | 0.0656 | 0.583 |
| 1.0 | 8 | 459 | 42.3 | 0.0513 | 0.452 |
| 1.5 | 8 | 699 | 65.3 | 0.00942 | 0.638 |
| 2.0 | 7 | 891 | 87.2 | 2.03e-14 | 0.544 |
| 2.5 | 7 | 1.09e+03 | 115 | 0.00654 | 0.467 |
| 3.0 | 6 | 1.33e+03 | 139 | 2.15e-14 | 0.367 |
| 3.5 | 6 | 1.45e+03 | 136 | 2.22e-14 | 0.276 |
| 4.0 | 6 | 1.59e+03 | 162 | 2.13e-14 | 0.28 |
| 4.5 | 6 | 1.7e+03 | 205 | 2.29e-14 | 0.285 |
| 5.0 | 6 | 1.81e+03 | 272 | 0.00242 | 0.199 |
| 5.5 | 6 | 1.97e+03 | 307 | 2.14e-14 | 0.169 |
| 6.0 | 4 | 1.76e+03 | 323 | 2.13e-14 | 0.131 |
| 6.5 | 3 | 1.76e+03 | 413 | 0.00655 | 0.112 |
| 7.0 | 3 | 1.89e+03 | 476 | 1.85e-14 | 0.106 |
| 7.5 | 3 | 2.02e+03 | 508 | 0.0014 | 0.151 |
| 8.0 | 2 | 2.08e+03 | 521 | 0.0158 | 0.103 |
| 8.5 | 2 | 2.51e+03 | 711 | 0.0018 | 0.0734 |
| 9.0 | 1 | 2.87e+03 | 890 | 1.52e-14 | 0.0723 |

### Temperature by forecast lead

All four models use the same eight dates. RMSE is the square root of equally date-weighted MSE. Temperature differences in K and °C have the same numerical value.

| Lead (days) | Persistence RMSE (°C) | Original RMSE (°C) | Extended RMSE (°C) | GenCast RMSE (°C) | Original CRPS (°C) | Extended CRPS (°C) | GenCast CRPS (°C) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.182 | 2.063 | 2.053 | 0.722 | 0.965 | 0.956 | 0.349 |
| 3 | 2.989 | 3.169 | 3.137 | 1.023 | 1.501 | 1.471 | 0.474 |
| 5 | 3.305 | 3.762 | 3.709 | 1.340 | 1.799 | 1.750 | 0.607 |
| 10 | 3.620 | 5.185 | 5.052 | 2.050 | 2.638 | 2.503 | 0.903 |
| 15 | 3.724 | 6.714 | 6.441 | 2.467 | 3.790 | 3.528 | 1.068 |

Persistence holds initial weather fields fixed; the dry reference predicts zero rain. Deterministic references have zero spread and are not calibrated ensembles. Historical climatology defines REV zero.

</details>

Aggregate panels: [original comparisons](results.json), [extended comparisons](extended-results.json), [all four models at every lead](main-evaluation-by-lead.csv).

## Ablation: additional autoregressive training

Compare the original 562-update Seed 0 with a 580-update continuation from AR2 (update 556), retaining optimizer state and increasing AR3–AR8 from one to four updates each. Both checkpoints use the same four held-out 2021 development dates, eight members, inputs, normalization and random keys. Positive percentages mean lower error.

| Variable | 1 d RMSE reduction | 3 d | 5 d | 10 d | 10 d CRPS reduction |
|---|---|---|---|---|---|
| 2 m temperature | 0.61% | 0.75% | 1.36% | 2.73% | 4.46% |
| Sea-level pressure | 1.08% | 4.52% | 5.44% | 6.40% | 4.53% |
| 500 hPa geopotential | 0.95% | 3.09% | 3.84% | 3.50% | 3.62% |
| 10 m wind speed | 0.34% | 0.94% | 1.71% | 2.47% | 2.84% |
| 6 h precipitation ending at lead | 0.06% | 0.02% | 0.04% | 0.42% | 1.24% |

**Additional AR-stage training improves all 20 aggregate variable/lead combinations in both RMSE and CRPS.** Temperature, pressure, geopotential and wind RMSE improve on all four dates at every reported lead. Individual-date precipitation results are mixed at some short leads.

The primary endpoint, 10-day wind-speed RMSE, decreases from **9.927 to 9.682 m/s (2.47%)**. Ten-day temperature RMSE decreases from **4.913 to 4.779°C**, and pressure from **15.31 to 14.33 hPa**. The gains support additional AR training as a useful intervention, but long-range errors remain large. This development diagnostic is distinct from the full 2022 test comparison above. The full comparison retains the original checkpoint alongside the extended model and baselines.

Continuation training consumed **46.37 RTX PRO 6000 GPU-hours** (four GPUs × 11 h 35 min 32 s), outside the original training budget. Four development dates do not establish statistical significance or full-test generalization. The experiment changes both rollout-training exposure and total optimization compute, so it does not isolate an AR-specific causal explanation.

## Compute cost

GPU-hours are GPU count × elapsed hours; GPU models are listed separately because their performance differs.

| Workload | GPU | GPU-hours | Basis |
|---|---|---:|---|
| Main recipe, 580 updates | RTX PRO 6000 | **87.51** | Composed measured prefix + continuation; 4 GPUs × 21.88 h |
| Train seed 0, 562 updates | RTX PRO 6000 | 59.45 | Measured: 4 GPUs × 14.86 h |
| Train seed 1, 562 updates | RTX PRO 6000 | 61.44 | Measured: 4 GPUs × 15.36 h |
| AR continuation paired inference and scoring | RTX A6000 | **1.63** | Measured GPU allocation: 1 h 37 min 57 s |
| Additional AR3–AR8 continuation | RTX PRO 6000 | **46.37** | Measured; outside original budget |
| Original training total, two seeds | RTX PRO 6000 | **120.89** | Measured training total |
| WN2 comparison inference, per seed | RTX A6000 | **30–40** | Estimated |
| WN2 comparison inference, both seeds | RTX A6000 | **60–80** | Estimated |
| GenCast comparison inference | RTX PRO 6000 | **330–400** | Estimated |
| Original independent submission inference | RTX A6000 | **17.87** | Measured allocation, including output validation |
| Original independent submission inference | RTX PRO 6000 | **16.01** | Measured allocation, including output validation |

The main recipe costs about **47% more training compute** than original Seed 0. Its 87.51 GPU-hour estimate sums the measured shared prefix (41.14 GPU-hours through AR2) and extended AR3–AR8 continuation (46.37 GPU-hours), retaining optimizer state. It is not a fresh uninterrupted 580-update run. The training envelope is **24 hours × 4 RTX PRO 6000 = 96 GPU-hours**, plus 30 minutes for packaging; evaluation is separate.

The extended checkpoint's complete independent evaluation used **5.04 RTX PRO 6000 GPU-hours + 13.81 RTX A6000 GPU-hours**, and **347.05 allocated CPU-hours** for output validation, scoring and paired comparisons. GPU allocations include interrupted work. CPU output validation runs after GPU inference; these costs exclude training and image builds.

Evaluation estimates cover the additional **8-date × 56-member × 15-day comparison**, or 448 trajectories per model. Measured sustained throughput extrapolates to approximately 26 A6000 GPU-hours per WN2 seed and 326 RTX PRO 6000 GPU-hours for GenCast; the ranges allow for initialization and throughput variation. These are throughput-based estimates, not measured full-run allocations. Metric scoring runs on CPUs and is excluded. Training costs exclude profiling, packaging and evaluation; estimates exclude development retries.

The independent submission inference allocations are additional to the research-comparison inference above. They cover all allocated GPU time, including output validation and interrupted work; CPU scoring and the separate one-member smoke are excluded.

CPU scoring for the original independent submission evaluation uses **390.58 allocated CPU-hours** and **4 h 39 min 46 s** of wall time. GPU inference and CPU scoring run as separate phases.

## Limitations

1. **Limited test coverage.** Eight fixed 2022 initializations cover about 0.55% of the paper's full-year initialization count. They provide limited seasonal and extreme-event coverage and cannot establish full-year skill or paper-level statistical significance. The 0.25° grid, 56 members and 15-day horizon are retained.
2. **Restricted training data and compute.** Training uses 32 sampled windows from 2017–2020 and a 580-update curriculum. This budget tests learning under constrained resources; it does not measure the architecture's performance after training on the full archive or at the paper's scale.
3. **Different temporal split and reference models.** The track trains through 2020, develops on 2021 and tests on 2022; the paper's final model trains through 2022 and tests on 2023. GenCast Operational <2022 has a later training cutoff than the candidate and is not the paper's exact baseline asset. Public WeatherNext2 <2025 weights overlap this test period and cannot provide a held-out reference. These comparisons do not reproduce the paper's score table or isolate architecture effects.
4. **Single-model ensemble.** Each candidate checkpoint supplies all 56 members, whereas the paper combines four trained models with 14 members each. The track therefore does not assess the contribution of diversity across independently trained models.
5. **Approximate operational inputs.** ERA5 sea-surface temperature and WN2 100 m wind inputs supplement the available HRES fields. This input configuration differs from the paper's operational setup, limiting conclusions about operational equivalence.

References: [FGN paper](https://arxiv.org/html/2506.10772v1), [WeatherBench data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), [official model versions](https://github.com/google-deepmind/weathernext#provided-pretrained-models).

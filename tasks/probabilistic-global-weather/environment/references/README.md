# References

Material we have **not** verified in this setup.

- WeatherNext 2 (Google DeepMind): the model family of the starter; the task uses the native
  WeatherNext code at `89c4b2a77a1c57b328b909c575550fd2e5aadc9c` (in the image under
  `/opt/weathernext`). This task trains a reduced recipe from scratch; it does not reproduce the
  published score table.
- GenCast (Price et al., Nature 2025): the probabilistic reference the task author compared against.
- WeatherBench-X at `88af2fde7a78f13b77496a2abb8ca88905fa5e95` (`/opt/weatherbenchX`): the metric
  implementations (empirical CRPS, SEEPS, spectra, pooling) used by the development and
  independent evaluations.
- ERA5 (Hersbach et al., 2020), WeatherBench 2 HRES-fc0, and NOAA IBTrACS v04r01 for cyclone tracks.

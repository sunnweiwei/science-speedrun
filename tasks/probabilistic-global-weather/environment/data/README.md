# WN2 data

```text
/workspace/data/
  README.md
  training/                         read-only: the fixed 2017–2020 training windows
    case-000/ ... case-031/         32 frozen training windows
    statistics/                     training-only normalizers: manifest.json, era5_1deg_12h/,
                                    era5_1deg_6h/, era5_025deg_6h/, hres_025deg_6h/
  development/                      read-only: four visible 2021 dates for model selection
    inputs/inputs-000.nc ... inputs-003.nc     two six-hourly context frames per date
    weather/{truth.zarr,status.json,protocol.json,cases/}
    precipitation/{truth.zarr,status.json,protocol.json,cases/}
```

Training uses only the 2017–2020 windows and the training-only statistics (normalizers and cyclone
labels). The 2021 development dates are for model selection, never for fitting or normalization.
The eight-date 2022 test panel (56 members, 30 twelve-hour leads through 15 days) is not in this
container; the verifier runs the package on it in an offline namespace that exposes only the
context frames and the model, never truth or training data. Write checkpoints, caches and forecasts
under `/workspace/research`.

Sources: ERA5 and ARCO-ERA5, WeatherBench HRES-fc0, and a frozen NOAA IBTrACS snapshot
(2026-09-06) for cyclone labels. Preserve each source's attribution and terms; the code's Apache-2.0
license does not relicense the data. How the resources were prepared: `environment/data/MATERIALIZE.md`
in the task repository.

---
license: other
license_name: weather-source-licenses
license_link: https://huggingface.co/datasets/sanxing/ai4sci-probabilistic-global-weather/blob/main/LICENSES.md
language:
  - en
tags:
  - weather
  - climate
  - weathernext2
  - era5
  - ai4science
  - benchmark
size_categories:
  - n<1K
---

# AI4Science probabilistic global weather: prepared WN2 resources

Frozen, ready-to-use data for the native WeatherNext 2 track in
[T0-RSI/ai4sci-tasks](https://github.com/T0-RSI/ai4sci-tasks/tree/main/probabilistic-global-weather).
This release contains approximately 205.2 GiB of prepared NetCDF/Zarr resources.
It avoids rebuilding training examples, normalizers and evaluation climatologies
from large, rolling upstream archives. It contains no pretrained model weights,
candidate predictions or experiment logs.

## Contents and intended use

- 32 fixed training windows from 2017–2020, cyclone labels and training-only
  normalizers; train the native model from random initialization.
- Four visible 2021 development contexts and weather/precipitation truth stores.
- Eight fixed 2022 evaluation dates, their contexts and scoring resources,
  including REV/SEEPS climatologies, spherical pooling, spectral references and
  cyclone tracks. Evaluation uses 56 ensemble members and 30 twelve-hour leads
  through 15 days on a 721×1440 grid.
- The exact NOAA IBTrACS snapshot acquired on 2026-09-06, with SHA-256
  `5584637db168ddd65e115114888ae2bea9587d0439954c2bde776703525005b5`.

REV uses 1993–2016 ERA5; SEEPS uses 1980–2015 ERA5. Resource creation and source
identities are described in `provenance/`. This is a deliberately reduced
benchmark subset, not the full published WeatherNext 2 evaluation.

## Download and layout

Use `probabilistic-global-weather/environment/data/materialize.sh TARGET`
from the linked task repository. It downloads anonymously at the checked-in
immutable HF revision, verifies SHA-256, and safely extracts the selected scope.
Install `huggingface_hub` on the operator host first. `--scope development`
downloads training and visible development resources; `--scope evaluation`
downloads the independent evaluation bundle. Default scope is `all`.

The dataset also provides a standalone downloader, so downloading the resources
does not require access to the task's GitHub repository:

```bash
python3 -m pip install 'huggingface_hub>=1.0,<2'
hf download sanxing/ai4sci-probabilistic-global-weather --repo-type dataset \
  --include download_hf.py download-source.json --local-dir weather-download
python3 weather-download/download_hf.py /absolute/path/to/wn2-data
```

The downloaded source configuration pins the data commit and manifest checksum;
changing the repository's default branch does not change those resource bytes.

`release.json` inventories each tar shard and every extracted file by path,
byte count and SHA-256. Shards preserve existing NetCDF/Zarr bytes and relative
paths; use xarray with the pinned task runtime after extraction. This is a
file-based dataset, not a table intended for `datasets.load_dataset`.

```text
TARGET/
  IBTrACS.since1980.v04r01.nc
  training-2017-2020/{case-000/...case-031/,statistics/}
  development-inputs-2021/
  development-weather-2021/
  development-precipitation-2021/
  evaluation-resources/
```

Set `TASK_DATA_DIR=TARGET` for the existing Compose mounts. Download on the
operator host before starting offline training or inference. Allow about
220.4 GB for extracted data plus one archive (up to about 8.6 GB), and additional
space for images, training artifacts and forecasts. Full evaluation outputs
can require much more disk than the resource release.

## Benchmark boundaries

The agent receives training and visible development mounts only. The verifier
receives the dedicated evaluation bundle; candidate inference receives only
context, runtime and its submitted model. Do not mount this repository's entire
extracted root into the agent or candidate inference process. Public retrospective
test truth is not secret and must not be used for training or model selection.

The sources were subsetted and transformed using the pinned track preparation
code: context/target construction, resolution changes for training, derived
fields, cyclone labels and calibration, normalization, climatology summaries,
and isolated-cell/SST masking repairs described by the task. This distribution
preserves those prepared bytes rather than recomputing them at download time.

## Sources, licenses and citations

See [LICENSES.md](LICENSES.md) and `licenses/` for source-specific terms.
The task code's Apache-2.0 license does not relicense these data.

- ERA5: Copernicus Climate Change Service / ECMWF, distributed through
  [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
  and [ARCO ERA5](https://github.com/google-research/arco-era5).
  Contains modified Copernicus Climate Change Service information (2026).
  Neither the European Commission nor ECMWF is responsible for any use that
  may be made of the Copernicus information or data it contains.
- HRES-fc0: Copyright © 2026 European Centre for Medium-Range Weather Forecasts
  (ECMWF), source https://www.ecmwf.int. Published under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Subsets and derived
  resources have been modified as described above. ECMWF does not accept any
  liability whatsoever for any error or omission in the data, their availability,
  or for any loss or damage arising from their use.
- NOAA NCEI [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive),
  including contributing tropical cyclone agencies. Cite the dataset
  [DOI: 10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16), indicate access
  on 2026-09-06, and retain its use constraints and attribution.
  Also cite Knapp et al. (2010), *Unifying tropical cyclone best track data*,
  [DOI: 10.1175/2009BAMS2755.1](https://doi.org/10.1175/2009BAMS2755.1).
- ARCO ERA5: Carver and Merose (2023), *An Analysis-Ready Cloud-Optimized
  Reanalysis Dataset*, [AMS presentation](https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842).
  ERA5: Hersbach et al. (2017), *Complete ERA5: Fifth generation of ECMWF
  atmospheric reanalyses of the global climate*, Copernicus Climate Change
  Service Data Store.
- WeatherBench 2: Rasp et al., *A Benchmark for the Next Generation of
  Data-Driven Global Weather Models*,
  [DOI: 10.1029/2023MS004019](https://doi.org/10.1029/2023MS004019).

Prepared track adaptation: sc829 / sanxing. Upstream organizations do not
endorse this task or its reduced evaluation protocol.

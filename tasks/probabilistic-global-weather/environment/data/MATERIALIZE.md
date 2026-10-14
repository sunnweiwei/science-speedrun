# Materializing the WN2 resources (operators)

Controller-side; the agent sees `README.md` of this directory only. The development container mounts
`training-2017-2020/` and the three `development-*-2021/` trees read-only under `/workspace/data`
(`TASK_DATA_DIR`); the verifier mounts `evaluation-resources/` (`TASK_PRIVATE_DIR`, usually the same
root).

Download the frozen prepared resources from the public
[Hugging Face dataset](https://huggingface.co/datasets/sanxing/ai4sci-probabilistic-global-weather)
on the operator host before allocating GPUs. No HF account or token is needed.
The agent receives only training and visible 2021 development mounts. The
verifier receives the dedicated `evaluation-resources` bundle; its inference
namespace exposes context and the portable model, never truth or training data.

```bash
python3 -m pip install 'huggingface_hub>=1.0,<2'
export TASK_DATA_DIR=/absolute/path/on/large-disk/wn2-data
bash probabilistic-global-weather/environment/data/materialize.sh "$TASK_DATA_DIR"
```

The checked-in `download-source.json` pins the HF commit and release manifest
SHA-256. Downloads resume by shard, archives and extracted files are verified,
and original NetCDF/Zarr paths and bytes are preserved. Rerunning checks existing
files and downloads only missing/corrupt shards. Archives are removed after
successful extraction unless `--keep-archives` is set. Allow about 220.4 GB for
resources plus one archive (up to about 8.6 GB), in addition to runtime artifacts
and forecast output storage. No upstream preparation or model environment is
needed for this download.

- `--scope development`: training, normalizers, frozen IBTrACS and visible 2021 data.
- `--scope evaluation`: only the independent evaluation resource bundle.
- `--scope all` (default): both scopes.
- `--verify-only`: rehash the selected downloaded resources offline.

Release metadata, source notices and the completion receipt are stored under
`.hf-release/`; these are operator files and are not mounted into inference.
The source materializer remains available for maintainer rebuilds with
`materialize.sh --upstream ...`, as documented below.

```text
TASK_DATA_DIR/
  IBTrACS.since1980.v04r01.nc
  training-2017-2020/case-000/ ... case-031/
  training-2017-2020/statistics/{manifest.json,era5_1deg_12h/,era5_1deg_6h/,era5_025deg_6h/,hres_025deg_6h/}
  development-inputs-2021/inputs-000.nc ... inputs-003.nc
  development-weather-2021/{truth.zarr,status.json,protocol.json,cases/}
  development-precipitation-2021/{truth.zarr,status.json,protocol.json,cases/}
  evaluation-resources/
    data/{inputs/,weather/,precipitation/,ready.json}
    rev-climatology/
    seeps/climatology/
    pooling/
    spectra/truth/
    tracker/{truth/,named-truth.json}
    training-scales.nc
    resources.json
```

The test bundle has eight preselected 2022 dates, 56 members and 30 twelve-hour
leads through 15 days. Do not mount an experiment directory containing model
predictions, checkpoints or logs in place of this resource-only bundle.

## Frozen IBTrACS snapshot (upstream rebuild only)

The HF release already includes this snapshot. For an upstream rebuild, it is
also distributed as a
[data-only repository release](https://github.com/T0-RSI/ai4sci-tasks/releases/tag/weather-ibtracs-20260906).
The repository is currently private: use GitHub authentication with repository
read access. Use the HF release for anonymous access.

For a fresh resource directory, download on the operator host:

```bash
export TASK_DATA_DIR=/absolute/path/on/large-disk/wn2-data
mkdir -p "$TASK_DATA_DIR"
gh release download weather-ibtracs-20260906 --repo T0-RSI/ai4sci-tasks \
  --pattern IBTrACS.since1980.v04r01.nc --dir "$TASK_DATA_DIR"
```

The preparation helper checks SHA-256
`5584637db168ddd65e115114888ae2bea9587d0439954c2bde776703525005b5`.
This approximately 11 MB NetCDF is the unmodified NOAA snapshot downloaded on
2026-09-06. NOAA's `access/netcdf` URL is rolling even with the `v04r01` name;
a newer file must not silently replace these training labels. An existing local
copy or `--ibtracs-url` override must match the same digest.

## Maintainer rebuild directly from upstream

Run `python3 probabilistic-global-weather/environment/data/check_sources.py`
before image builds or large downloads. The ARCO policy allows changes only to
`last_updated`, `valid_time_stop` and `valid_time_stop_era5t`; all other metadata
must match the frozen schema, and stable coverage must include all of 2022.
The original exact metadata pin remains valid for existing receipts. See the
[standalone guide](../../docs/RUNNING.md) for the source prerequisite and commands.

Use the pinned Dockerfile-derived runtime and Apptainer, or provide the model
and evaluator Python environments explicitly. Preparation code is mounted from
the checkout and executes online; it is absent from the offline agent workflow.

```bash
bash probabilistic-global-weather/environment/data/materialize.sh --upstream \
  "$TASK_DATA_DIR" --image /absolute/path/to/tests.sif \
  --workers 2 --compute-workers 1 --job-timeout 172800 \
  --ibtracs "$TASK_DATA_DIR/IBTrACS.since1980.v04r01.nc"
```

Use the verifier image (`tests.sif`), which includes cyclone tracker binaries.
Build it with the [standalone commands](../../docs/RUNNING.md) or the optional
site-specific [Slurm wrapper](../../slurm/wn2_build.sbatch).
Without `--image`, set `--eval-python`, `--model-python` and `--wbx-source` to
pinned local environments. Download the snapshot before running inside a
container; GitHub credentials are not mounted into it.

- `--scope development`: training windows, cyclone labels, training-only
  normalizers and the four visible 2021 contexts/truth stores.
- `--scope evaluation`: the single eight-date evaluation bundle; requires
  existing training statistics and the frozen IBTrACS file.
- `--scope all` (default): both scopes, in dependency order.

Evaluation preparation builds the common weather/precipitation data, empirical
REV thresholds, historical SEEPS climatologies, spherical pooling geometry,
HRES spectral references and observed cyclone history. The large historical
precipitation cache stays under `.materialize/`; it is never mounted into the
verifier. Actual preparation time and scratch storage depend on upstream
throughput. It can take much longer than training and is outside that budget.
Provision substantial additional scratch space for upstream preparation.
The default windowed evaluator provisions 1 TB for resources, state and forecasts;
retaining all forecasts or splitting forecast/score phases requires 8 TiB.
Neither number is the dataset download size.

Upstream weather metadata identities are defined in
[sources.lock.json](sources.lock.json): exact metadata for ERA5/HRES and a
fixed schema with three explicitly allowed rolling attributes for ARCO.
These checks do not guarantee immutable upstream chunks. Materialized arrays receive inventories, and changed completed sources
fail instead of being silently replaced. Preparation resumes completed cases;
interrupted steps retain logs under `.materialize/`. The fixed training cutoff
is 2020. REV uses 1993–2016 ERA5, and SEEPS uses 1980–2015 ERA5.

## Offline verification of upstream preparation receipts

```bash
bash probabilistic-global-weather/environment/data/materialize.sh --upstream \
  "$TASK_DATA_DIR" --verify-only --workers 2 \
  --eval-python /absolute/path/to/eval/bin/python \
  --model-python /absolute/path/to/model/bin/python \
  --wbx-source /absolute/path/to/weatherbenchX --job-timeout 172800
```

Verification rehashes prepared bytes and can take significant I/O time. It does
not download or allocate GPUs. `evaluation-resources/resources.json` records
the complete prepared bundle. A full clean-room reconstruction and Docker/Compose
execution must not be inferred from local resource availability alone.

Sources: [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive),
ERA5/ARCO ERA5 and WeatherBench HRES-fc0. Preserve each source's attribution and
terms; the code's Apache-2.0 license does not relicense the data.

# Prepared Hugging Face dataset feasibility

Assessed 2026-09-19 against the merged task. Publishing the prepared resources
is technically feasible and removes upstream preparation from ordinary setup.
The prepared resources have now been published publicly at
[sanxing/ai4sci-probabilistic-global-weather](https://huggingface.co/datasets/sanxing/ai4sci-probabilistic-global-weather).
The release contains 47 tar shards, 35,273 files and 220,378,863,635 extracted
bytes. See [download-source.json](download-source.json) for the immutable data
revision and manifest checksum, [current setup](README.md) for usage, and
[release validation](../../docs/validation/hf-release.json) for the measured checks.
The sections below preserve the original assessment and design rationale.

## Measured local resources

The directory inventory is recorded in [hf-feasibility-inventory.json](hf-feasibility-inventory.json).
Sizes are existing file bytes, before archive overhead or additional compression.

| Component | GiB | Files |
| --- | ---: | ---: |
| Training directory, including optional baseline climatology | 126.320 | 1,761 |
| Development contexts | 1.141 | 8 |
| Development weather truth | 28.725 | 13,167 |
| Development precipitation truth | 0.362 | 189 |
| Dedicated evaluation resources | 54.800 | 20,195 |
| Frozen IBTrACS snapshot | 0.010 | 1 |
| Total inspected | 211.358 | 35,321 |

The canonical release should include only training `case-000` through `case-031`,
`statistics/`, the three development directories, IBTrACS, and
`evaluation-resources/`. The extra training `climatology/` is a baseline resource,
not required by the starter or materializer: excluding its 48 files (6.114 GiB)
gives approximately **205.244 GiB / 220.379 GB and 35,273 files**.
The evaluation receipt records `passed` and 20,194 inventoried files, plus the
receipt itself. No file or directory symlinks were found in training or evaluation.
This inventory does not establish fresh byte integrity; rehash before release.

Do not archive the enclosing experiment directory. Exclude checkpoints,
predictions, logs, profiling runs, images, upstream historical precipitation
caches and `.materialize/`. The original 8 TiB evaluator provision included all
generated forecasts and scoring workspace. The bounded pipeline now provisions
1 TB (four concurrent cases) using lossless continuation and window cleanup; retaining all forecasts still requires 8 TiB.
Neither provision is the dataset download size.

## Delivery design

Use the existing operator-download/offline-mount pattern:

- [HydroGym](../../../fluid-flow-control/environment/data/materialize.py) has a separate
  [download source](../../../fluid-flow-control/environment/data/download-source.json)
  with a fixed repository and commit SHA, selected downloads and verification.
- [FoldBench](../../../protein-folding-prediction/README.md#environment-and-data-preparation)
  similarly separates HF materialization from read-only runtime mounts; its
  current main-branch instructions still require release defaults to be filled.

Keep native NetCDF/Zarr bytes and paths. There is no reason to convert weather
arrays to tabular Hugging Face `datasets` rows. Use `huggingface_hub` for transport
and retain the existing xarray readers after extraction.

Publish independent tar shards targeting 4–8 GiB each, grouped into training,
development and evaluation scopes. Each training case is already roughly
3.75 GiB and is a natural shard. Split larger Zarr trees across bounded archives
without modifying their member files. Avoid recompressing already compressed
arrays unless a sample shows worthwhile savings. At assessment time, archive size and transfer performance had not been
measured; the release record now provides the actual shard counts and sizes.

Add a release manifest containing each shard's SHA-256, bytes, scope and safe
relative member paths, plus per-file hashes and the current protocol/source
identities. Pin the final HF commit SHA in a checked-in download-source file.
Download on the operator host with scope selection and resumable transfers;
verify hashes, safely extract into staging, verify the resource inventory, then
publish a ready marker. Reject path traversal, links and unexpected files.
Keep upstream preparation available as a maintainer rebuild path.

Restore exactly the existing `TASK_DATA_DIR` layout. Existing Compose mounts
already separate training/development from evaluation; keep those boundaries.
The agent must never receive the evaluation subtree or the operator's HF token.
Public hosting does not make retrospective test truth secret; the task already
documents this and relies on offline inference isolation. If access-controlled
test distribution is wanted, place evaluation shards in a separate private repo.

## Hosting and source terms

The [Hub storage guidance](https://huggingface.co/docs/hub/storage-limits)
recommends fewer than 100,000 repository files. Sharding makes this release much
smaller in file count. Public free storage is best-effort; the free private
allowance is 100 GB, less than this complete release. Check the destination
account's available quota. Do not assume unlimited free hosting.
Use the resumable
[large-folder upload workflow](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder)
and freeze a revision only after all shards and the manifest are verified.

The [WeatherBench data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
directs users to each dataset's bucket LICENSE. The currently served
[HRES-fc0 license](https://storage.googleapis.com/weatherbench2/datasets/hres_t0/LICENSE)
explicitly allows redistribution under CC BY 4.0 with ECMWF attribution,
disclaimers and modification notices. The
[ERA5 license](https://storage.googleapis.com/weatherbench2/datasets/era5/LICENSE)
allows redistribution under the Copernicus terms with attribution and notices.
Preserve the [ARCO ERA5](https://github.com/google-research/arco-era5#license)
and [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
source notices and citations as well. Ship the actual source licenses and a
dataset card describing selection, transformations, years and evaluation use;
do not label the whole data release with the code's Apache-2.0 license.

## Original release checklist

1. Revalidate the selected local resources and freeze their complete manifest.
2. Build bounded shards and exercise extraction/corruption checks on a small sample.
3. Establish the destination HF account/repository, visibility and storage quota.
   Authentication has now been verified for `sanxing` using a locally cached
   fine-grained token. The account reports no PRO subscription. Destination
   repository write access was subsequently confirmed by the complete public
   upload. The local
   Zsh configuration overrides `HF_HOME`, so explicitly select the credential
   when it resides outside that configured directory.
4. Upload, pin the final commit, and implement the default HF download path.
5. Validate public anonymous downloads, extraction and existing resource checks.
   Release validation rehashes the complete original resource set, checks every
   remote shard SHA-256, and anonymously downloads/extracts representative shards
   from all seven component types. This is not a fresh full 205-GiB download or
   a new model evaluation; the validation record states that limit explicitly.

At 205.244 GiB, payload-only transfer takes roughly 37 minutes at a sustained
100 MB/s or 2.45 hours at 25 MB/s. These are arithmetic estimates, not measured
HF throughput; hashing, extraction and retries add time. Keeping archives and
extracted data simultaneously needs about 411 GiB before scratch space. Download
and extract one shard at a time to reduce peak local storage.

# Data

Two public Apache-2.0 sources from the GIFT-Eval authors, pinned to Hugging
Face commit SHAs in `sources.lock.json`, plus a pinned snapshot of the public
leaderboard used only as reference. Nothing is committed; `materialize.py`
downloads and verifies every file (LFS SHA-256, git blob SHA-1 for small
non-LFS files) and writes a receipt.

| id | repository @ revision | role | size |
|---|---|---|---|
| `gifteval` | `Salesforce/GiftEval` @ `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0` (2025-01-21) | evaluation: 23 datasets in 28 directories, 55 dataset-frequency variants, 97 configs, 144k series and, as published on the dataset card, 177M points; the verifier's only data | 1.59 GB, 175 files |
| `gifteval_pretrain` | `Salesforce/GiftEvalPretrain` @ `6830b624de7ed2b3d3e5b85bb6959d81dcc5d874` (2025-01-21) | pretraining corpus, 95 of 152 directories (below) | 273.21 GB, 1,662 files in the 95 directories (1,664 with the card and `.gitattributes`; of 975 GB, 6,682 files) |
| `leaderboard` | Space `Salesforce/GIFT-Eval` @ `a5019972f79e650b1ede252885df5f515fa115b8` (2026-09-15) | `results/<model>/all_results.csv` + `config.json` for 130 systems; the Seasonal Naive row is the task's normalizer, the rest fill the README tables | 3 MB, 267 files |

In this workspace, `/workspace/data` holds this file, the frozen split (`split.json`,
`split_smoke.json`) and the source lock, plus three read-only mounts:

```text
/workspace/data/gifteval_pretrain/<directory>/   the 95 allow-listed corpus directories (Arrow)
/workspace/data/valid/{input,truth}/              one validation window per GIFT-Eval series, in the contract's format
/workspace/data/materialize-receipt.json          what was downloaded, with checksums
```

`gifteval/`, which contains the test region, exists only in the verifier; `entrypoint.sh`
refuses to start if it is mounted here. How the data is prepared is in `MATERIALIZE.md` in the
task repository (not in this workspace).

## Layout

Both datasets are `datasets.save_to_disk` Arrow directories
(`data-NNNNN-of-NNNNN.arrow` + `dataset_info.json` + `state.json`); GIFT-Eval
datasets with several frequencies have one sub-directory per frequency
(`electricity/{15T,H,D,W}`). Rows are `item_id`, `start` (timestamp[s]),
`freq` (pandas alias), `target` (float32 list, or a fixed-size list of lists for
multivariate datasets). `datasets.load_dataset()` does not work on this
layout; use `datasets.load_from_disk(path)` or the pyarrow reader in
`evaluation/ts_io.py` (`read_arrow_dataset`), which needs no `datasets`
install. Stray `cache-*.arrow` files in the eval repo are listed in the lock
and verified but unused.

## Protocol (frozen in `split.json`)

Transcribed from gift-eval `src/gift_eval/data.py` @ `db743860` (the
leaderboard's code) and checked against it on real downloads
(`docs/validation/golden_gluonts_check.py`):

- horizon = term multiplier (short 1, medium 10, long 15) × base horizon of the
  frequency (M 12, W 8, D 30, H 48, minutes 48, seconds 60; M4 datasets:
  A 6, Q 8, M 18, W 13, D 14, H 48);
- rolling, non-overlapping test windows over the last 10% of every series:
  `windows = min(max(1, ceil(0.1 × min_series_length / horizon)), 20)`, one for
  M4; window k's cutoff is `T − horizon × windows + k × horizon`;
- medium and long terms exist for the 21 dataset-frequency variants the
  benchmark defines them for (frequencies of hourly and finer, plus the
  10-second bizitobs sets); 97 configs in total (55 short, 21 medium, 21
  long), 326,490 test instances, 40.7M label points;
- seasonality per frequency from gluonts' table (10S 360, 5T 288, 10T 144,
  15T 96, H 24, D 1, W 1, M 12, Q 4, A 1);
- validation = one window per series (151,170 instances), ending before the
  earliest test region of every config that shares that series. gift-eval's
  `validation_dataset` offset is the per-config upper bound; the tighter
  boundary is what keeps the agent-visible data free of held-out labels, since
  the terms of one series have test regions that start at different times. A
  series stored verbatim under two M4 directories counts as one series (its
  copies share the earliest boundary), and the 632 M4 series whose values hold
  another series' test label verbatim have no validation window
  (`make_cases.content_conflicts`; `docs/validation/valid_split_leakage.json`).

`split_smoke.json` is seven small configs (M4 single window, monthly,
daily with 20 windows, missing labels, weekly, two multivariate) for tests.

## Pretraining allow-list

The 95 directories below (273.21 GB, 63.2 G points). Until v2 the allow-list was 81 directories
(41.14 GB). v2 adds `buildings_900k` (63.0 GB, simulated building loads), `borg_cluster_data_2011`
(15.1 GB, Google cluster traces), nine ERA5 years 2010–2018 (116 GB) and three CMIP6 shards
(2000/2005/2010, 38 GB). Measured reason: a 50.9M TimesFM-3-architecture model trained on the 41 GB
corpus overfit within about 25,000 steps, and the enlarged corpus scored higher on the test split
(`docs/validation/corpus_scaling.json`). Still excluded: the other 51 climate shards (702 GB),
which would add volume but no new domain. The exact list and byte totals are in the lock, so runs are
comparable. The `fast` tier (64 directories under 200 MB, 1.25 GB) covers most of the domain
diversity for smoke runs. Series shorter than 64
observations or without a finite value are dropped by the starter's packer.

| directory | MB | tier | frequencies | series (≥64 points) | M points |
|---|---:|---|---|---:|---:|
| `BEIJING_SUBWAY_30MIN` | 22.6 | fast | 30T | 552 | 0.9 |
| `HZMETRO` | 1.5 | fast | 15T | 160 | 0.4 |
| `LOS_LOOP` | 28.4 | fast | 5T | 207 | 7.1 |
| `PEMS03` | 37.5 | fast | 5T | 358 | 9.4 |
| `PEMS04` | 62.6 | fast | 5T | 921 | 15.6 |
| `PEMS07` | 99.7 | fast | 5T | 883 | 24.9 |
| `PEMS08` | 36.4 | fast | 5T | 510 | 9.1 |
| `PEMS_BAY` | 67.8 | fast | 5T | 325 | 16.9 |
| `Q-TRAFFIC` | 1,059.1 | full | 15T | 45,148 | 264.4 |
| `SHMETRO` | 20.3 | fast | 15T | 576 | 5.1 |
| `alibaba_cluster_trace_2018` | 3,049.4 | full | 5T | 116,818 | 190.4 |
| `australian_electricity_demand` | 4.6 | fast | 30T | 5 | 1.2 |
| `azure_vm_traces_2017` | 10,632.1 | full | 5T | 159,472 | 885.5 |
| `bdg-2_bear` | 5.9 | fast | H | 91 | 1.5 |
| `bdg-2_fox` | 9.3 | fast | H | 135 | 2.3 |
| `bdg-2_panther` | 3.7 | fast | H | 105 | 0.9 |
| `bdg-2_rat` | 18.9 | fast | H | 280 | 4.7 |
| `bitcoin_with_missing` | 0.3 | fast | D | 18 | 0.1 |
| `borealis` | 0.3 | fast | H | 15 | 0.1 |
| `bull` | 11.5 | fast | H | 41 | 0.7 |
| `cdc_fluview_ilinet` | 1.3 | fast | W | 326 | 0.3 |
| `cdc_fluview_who_nrevss` | 0.7 | fast | W | 296 | 0.2 |
| `china_air_quality` | 137.8 | fast | H | 2,605 | 34.3 |
| `cif_2016_12` | 0.0 | fast | M | 48 | 0.0 |
| `cif_2016_6` | 0.0 | fast | M | 3 | 0.0 |
| `cockatoo` | 0.4 | fast | H | 1 | 0.0 |
| `covid19_energy` | 0.9 | fast | H | 1 | 0.0 |
| `covid_mobility` | 0.6 | fast | D | 362 | 0.1 |
| `elecdemand` | 0.1 | fast | 30T | 1 | 0.0 |
| `elf` | 0.1 | fast | H | 1 | 0.0 |
| `extended_web_traffic_with_missing` | 1,487.7 | full | D | 145,063 | 370.9 |
| `favorita_sales` | 560.2 | full | D | 111,840 | 139.2 |
| `favorita_transactions` | 0.3 | fast | D | 54 | 0.1 |
| `gfc12_load` | 12.6 | fast | H | 20 | 0.8 |
| `gfc14_load` | 0.1 | fast | H | 1 | 0.0 |
| `gfc17_load` | 2.3 | fast | H | 8 | 0.1 |
| `godaddy` | 1.1 | fast |  | 0 | 0.0 |
| `hog` | 10.1 | fast | H | 24 | 0.4 |
| `ideal` | 5.0 | fast | H | 217 | 1.3 |
| `kaggle_web_traffic_weekly` | 70.7 | fast | W-SUN | 145,063 | 16.5 |
| `kdd2022` | 189.1 | fast | 10T | 134 | 4.7 |
| `largest_2017` | 3,446.5 | full | 5T | 8,196 | 861.6 |
| `largest_2018` | 3,544.1 | full | 5T | 8,428 | 886.0 |
| `largest_2019` | 3,616.4 | full | 5T | 8,600 | 904.0 |
| `largest_2020` | 3,609.8 | full | 5T | 8,561 | 902.4 |
| `largest_2021` | 3,594.5 | full | 5T | 8,548 | 898.6 |
| `lcl` | 38.2 | fast | H | 713 | 9.5 |
| `london_smart_meters_with_missing` | 665.1 | full | 30T | 5,520 | 166.2 |
| `m1_monthly` | 0.2 | fast | M | 361 | 0.0 |
| `m1_quarterly` | 0.0 | fast | Q-DEC | 44 | 0.0 |
| `m1_yearly` | 0.0 | fast |  | 0 | 0.0 |
| `m5` | 234.8 | full | D | 30,490 | 58.3 |
| `monash_m3_monthly` | 0.6 | fast | M | 1,083 | 0.1 |
| `monash_m3_other` | 0.1 | fast | Q-DEC | 54 | 0.0 |
| `monash_m3_quarterly` | 0.2 | fast | Q-DEC | 86 | 0.0 |
| `nn5_daily_with_missing` | 0.3 | fast | D | 111 | 0.1 |
| `nn5_weekly` | 0.1 | fast | W-SUN | 111 | 0.0 |
| `oikolab_weather` | 3.2 | fast | H | 8 | 0.8 |
| `pdb` | 0.3 | fast | H | 1 | 0.0 |
| `project_tycho` | 5.6 | fast | W | 1,258 | 1.4 |
| `residential_load_power` | 1,752.0 | full | T | 813 | 438.0 |
| `residential_pv_power` | 1,504.1 | full | T | 699 | 376.0 |
| `rideshare_with_missing` | 3.5 | fast | H | 2,304 | 0.9 |
| `sceaux` | 0.1 | fast | H | 1 | 0.0 |
| `smart` | 0.4 | fast | H | 5 | 0.1 |
| `solar_power` | 29.6 | fast | 4S | 1 | 7.4 |
| `spain` | 0.6 | fast | H | 1 | 0.0 |
| `subseasonal` | 227.2 | full | D | 3,448 | 56.8 |
| `subseasonal_precip` | 39.1 | fast | D | 862 | 9.8 |
| `sunspot_with_missing` | 0.3 | fast | D | 1 | 0.1 |
| `taxi_30min` | 221.9 | full | 30T | 67,984 | 55.0 |
| `tourism_quarterly` | 0.2 | fast | Q-DEC | 342 | 0.0 |
| `tourism_yearly` | 0.1 | fast |  | 0 | 0.0 |
| `traffic_hourly` | 59.9 | fast | H | 862 | 15.0 |
| `traffic_weekly` | 0.4 | fast | W-SUN | 862 | 0.1 |
| `uber_tlc_daily` | 0.2 | fast | D | 262 | 0.0 |
| `uber_tlc_hourly` | 4.5 | fast | H | 262 | 1.1 |
| `vehicle_trips_with_missing` | 0.1 | fast | D | 226 | 0.0 |
| `wiki-rolling_nips` | 163.7 | fast | D | 47,675 | 40.6 |
| `wind_farms_with_missing` | 688.7 | full | T | 337 | 172.2 |
| `wind_power` | 29.6 | fast | 4S | 1 | 7.4 |
| `borg_cluster_data_2011` | 15,059.0 | full | 5T | 286,772 | 1,075.0 |
| `buildings_900k` | 63,020.1 | full | H | 1,795,256 | 15,728.0 |
| `cmip6_2000` | 12,680.0 | full | 6H | 434,176 | 3,170.0 |
| `cmip6_2005` | 12,680.0 | full | 6H | 434,176 | 3,169.0 |
| `cmip6_2010` | 12,680.0 | full | 6H | 434,176 | 3,170.0 |
| `era5_2010` | 12,883.6 | full | H | 368,640 | 3,220.0 |
| `era5_2011` | 12,883.6 | full | H | 368,640 | 3,221.0 |
| `era5_2012` | 12,883.6 | full | H | 368,640 | 3,220.0 |
| `era5_2013` | 12,883.6 | full | H | 368,640 | 3,220.0 |
| `era5_2014` | 12,883.6 | full | H | 368,640 | 3,221.0 |
| `era5_2015` | 12,883.6 | full | H | 368,640 | 3,220.0 |
| `era5_2016` | 12,883.6 | full | H | 368,640 | 3,221.0 |
| `era5_2017` | 12,883.6 | full | H | 368,640 | 3,220.0 |
| `era5_2018` | 12,883.6 | full | H | 368,640 | 3,221.0 |

Totals after the starter's packing rule: 941,809 series, 7.93 billion observations (a 30 GB float32 memmap; `docs/validation/task_trained_baseline.json`).

Two gaps between this corpus and the test configs are worth knowing before training. The six 10-second
configs (`bizitobs_application`, `bizitobs_service`) have almost no counterpart here: the only sub-minute
directories are `solar_power` and `wind_power`, one 4-second series each. And there is no yearly training
data under the starter's rule: `m1_yearly` and `tourism_yearly` hold no series of 64 or more observations,
so the default pack drops both while `m4_yearly` is scored.

## Leakage boundary

GiftEvalPretrain is advertised as having no series in common with GIFT-Eval,
and no dataset name is shared. **That claim does not survive a content check.**
Six directories were measured to contain GIFT-Eval evaluation series up to an
affine transform and are excluded from the allow-list
(`docs/validation/corpus_leakage_verified.json`, survey in
`docs/validation/corpus_overlap.json`):

| excluded directory | GIFT-Eval dataset | evidence |
|---|---|---|
| `pedestrian_counts` (12.5 MB) | `m4_hourly` | the same series plus a constant: correlation 1.000000 over all 748 observations, test horizon included, scale 1.0000, offset 10.000 |
| `weather` (171.8 MB) | `temperature_rain_with_missing` | 0.999494 over all 725 observations, scale 0.9984 |
| `fred_md` (0.3 MB) | `m4_monthly` | 0.999949 and 0.999755 over all 696 observations |
| `tourism_monthly` (0.4 MB) | `m4_monthly` | 0.999939 over a 175-observation window, scale 0.1002 |
| `monash_m3_yearly` (0.1 MB) | `m4_yearly` | 0.999994 over a 47-observation window, scale 1.9988 |
| `beijing_air_quality` (18.5 MB) | `kdd_cup_2018_with_missing` | precautionary: KDD Cup 2018 *is* Beijing air quality and the two share 46 anchor windows, though no whole series matched |

The cause is upstream: M4 and the Monash archive were assembled from
overlapping source pools, so a name-based exclusion cannot catch it. Removing
the six costs 204 MB of 41.3 GB, leaving 81 directories and 41.14 GB (v1).

**The v2 additions were checked by content.** Every 32-point window of the last 20 % of each GIFT-Eval
series (147,909 series, 26.8 M fingerprints) was z-normalized and rounded to 0.1. A corpus series
counts as a copy when 32 consecutive windows match, i.e. at least 63 aligned points up to scale and
offset. The check was first validated on controls:
- it flags `pedestrian_counts` (46 of 66 series match `m4_hourly`);
- it stays silent on `solar_power`.

On a 2 % sample of the 14 added directories (133,943 series) it found **no** copies
(`docs/validation/extra_dirs_leak_check.json`).

Still not verified: series-level disjointness for the near-name pairs (`weather`/`oikolab_weather` vs
`jena_weather`, `solar_power` vs `solar`, `traffic_hourly`/`PEMS_BAY`/`LOS_LOOP`
vs `LOOP_SEATTLE`/`SZ_TAXI`/`M_DENSE`, `m1_*`/`monash_m3_*`/`tourism_*` vs
`m4_*`, `elecdemand`/`london_smart_meters` vs `electricity`); these are
different datasets by provenance. Corpora that do overlap with GIFT-Eval test
data (LOTSA, chronos_datasets real configs, Time-300B, Timeseries-PILE,
Monash) are excluded by rule, and the released `kernel_synth_1m` file is
excluded for lack of a per-config license (agents regenerate synthetic data
themselves; `baseline/starter/synth.py`).

## License note

Both dataset cards declare Apache-2.0; both READMEs also carry Salesforce's
"research purposes only in support of an academic paper" ethical-considerations
paragraph. The task is research use. The card's Apache-2.0 tag does **not**
describe every constituent of the pretraining corpus: `largest_*` (LargeST) is
CC BY-NC 4.0 and is 43.1% of the allow-listed bytes, and about a quarter more
carries Kaggle, research-only or registration-gated terms. Nothing is
redistributed here, so those terms bind the user directly;
`../../UPSTREAM_LICENSES.md` has the table and the two options for tightening
the allow-list. `docs/UPSTREAM_LICENSES.md` in the task repository
collects every upstream license, including the non-commercial license of the
TimesFM-3 checkpoint, which is downloaded only by maintainers for the replay
and never redistributed.

## Why this corpus is not republished

`pde-surrogate` publishes a pinned data release on the Hub. This task does not,
and the reason is worth stating so nobody adds one for symmetry.

Every byte of the permitted corpus is **byte-identical to upstream**
`Salesforce/GiftEvalPretrain` at the pinned revision. What is task-specific is
which 95 of its 152 directories are permitted, and that is already pinned here
in `sources.lock.json` with a SHA-256 per file. Uploading 273 GB would duplicate
Salesforce's data to add nothing a consumer cannot already verify.

`pde-surrogate` is different because its split *creates* files: the corpus there
stores training and held-out trajectories in the same HDF5, so the release
carries truncated heads and held-out tails that exist nowhere upstream. Here the
corpus and the evaluation data are separate repositories to begin with.

Two things were checked before concluding this, both recorded in
`docs/validation/republication_readiness.json`:

- **Nothing in the corpus describes the held-out windows.** `pde-surrogate` found
  that PDEArena ships `pde_test_seed_*.yaml`, the generator seeds for its test
  trajectories, and nearly published them. The equivalent search here over all
  326 corpus files returns zero matches.
- **A release would have to be labelled CC BY-NC 4.0**, because LargeST is 43.1%
  of the allow-list and is non-commercial. Four Kaggle-sourced directories worth
  1.9% remain unresolved, their competition rules being login-walled.

If a release is ever wanted anyway — for one-step reproduction rather than for
the bytes — `build_release.py` produces it, and `tests/test_release.py` enforces
that the six leaking directories can never enter one.

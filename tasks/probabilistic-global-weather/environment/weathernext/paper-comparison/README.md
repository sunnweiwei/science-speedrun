# Paper-method subset comparison

This operator workflow reruns a published Operational GenCast baseline and our
trained WN2 models on the same eight preselected 2022 initialization dates.
This is the same frozen protocol used by formal submission evaluation. None of
these baseline tools enter the agent starter or supply pretrained weights to
candidate training. See the [current results](../docs/validation/results-summary/README.md)
and [delivery acceptance status](../docs/READINESS.md).

## Frozen forecast contract

- Eight dates, two per quarter, selected before seeing comparison scores.
- Native 0.25 degree grid; 13 pressure levels; 56 members; 15-day horizon.
- Compare weather at common 12-hour lead times, verified against HRES-fc0.
- Verify 12-hour precipitation against the sum of two ERA5 6-hour intervals.
  GenCast predicts this directly; WN2 must sum consecutive native 6-hour outputs.
- Context contains -12, -6 and 0 hour frames. GenCast uses -12 and 0;
  WN2 uses -6 and 0. Future templates contain no verification observations.
- Inference RNG depends on initialization, member identity and seed, independent
  of worker ordering. Completed member trajectories have checksummed inventories.

The baseline is `GenCast 0p25deg Operational <2022`, trained through 2021.
`baseline-assets.lock.json` pins public GCS object generations and SHA-256 values.
The runner retains its full published 20-noise-level sampler. GPU attention uses
Google's supported `triblockdiag_mha` implementation in place of TPU Splash.

## Reproduction

Run `download_baseline.py --output ASSETS` with standard Python. Run
`prepare.py --output DATA --workers 4` in the pinned evaluation environment.
`predict_gencast.py` runs in the pinned model image, with only the context input
folder, assets and output mounted. `gencast.sbatch` launches eight independent
member workers per allocation; workers claim unfinished members under filesystem locks, so additional suitable
GPU allocations can join without repartitioning RNG streams. Each member produces all 30 forecast leads.

## Comparability and unfinished work

The method reference is [the FGN paper](https://arxiv.org/html/2506.10772v1),
with spatial pooling defined in [GenCast Appendix E.6](https://arxiv.org/html/2312.15796v2).
The complete seed-0 and GenCast comparison includes empirical CRPS,
ensemble-mean RMSE, spread/skill, spherical average/max pooling, derived fields,
REV, SEEPS, spectra and cyclone diagnostics on all eight dates. Both ddof=0
and ddof=1 ensemble variances are retained explicitly. Simple-baseline native
scores are complete; their pooled diagnostics are still being computed.
Completed research scores are distinct from replaying a submitted artifact
through the final independent container entry point.

Known differences from the published experiment must accompany every report:

1. The paper tests 2023; this subset tests 2022 because the public HRES-fc0 archive
   used here does not provide 2023. This is not a reproduction of its score table.
2. These public GenCast weights are an earlier Operational checkpoint, not the
   exact checkpoint used for the paper's 2023 comparison. Historical forecast
   access requires authorization. Baseline training ends in 2021 while candidate
   training ends in 2020; both precede evaluation, but training budgets differ.
3. Public HRES lacks SST and ancillary inputs. Both models receive the same ERA5
   SST; WN2 additionally requires ERA5 100m winds. Operational input availability
   is therefore not reproduced exactly.
4. The paper's FGN ensemble combines four independently trained models, 14
   members each. The current track has two independent training runs. A
   56-member evaluation of either run is a single-model comparison, not that
   four-model ensemble; do not relabel the two runs as four independent models.
5. Eight dates cannot support the same significance claims or rare-event
   precision as the full-year experiment. Report paired case-level scores and
   the sample count, and do not import the paper's baseline numbers or claims.

Current assets/data preparation and baseline inference do not establish that a
candidate has passed final training, evaluation or repository release gates.

`predict_wn2.py` consumes only a checksummed final HRES AR8 package and retains
its native implementation and RNG rules. It emits common 12-hour leads and
aggregates consecutive 6-hour precipitation intervals. The complete seed-0 forecast and score results are linked above.

`dispatch_wn2.py` checks for the immutable `submission-cold` package every
30 minutes. It verifies the inventory, final HRES AR8 report and weight binding
before requesting GPUs. This package contains the completed final weights;
formal submission acceptance separately replays the package through `infer.sh`. The dispatcher requires explicit `--partition` and `--account` choices and
accepts `--gpus`, `--cpus`, `--memory`, `--time-limit` and `--exclude` resource
constraints. `--seeds 0` prioritizes the first model; `--worker-base` and a
separate `--dispatch-directory` support a new allocation while preserving
previous receipts and completed forecasts. Use the same frozen prediction
source and final package when resuming. No GPU inference job is submitted
before a final package exists. A requested allocation is not a capacity guarantee.

`wn2.sbatch` first runs one worker and gates subsequent workers on an actual
6-hour forecast. A final seed-0 checkpoint completed a full 15-day member on
A6000, with observed device usage 34,375 MiB and steady six-hour steps around
2.75 seconds including output I/O. Subsequent workers now start concurrently
after the first worker passes; the already-running seed-0 source-v1 retains
its original sequential startup. This reduces repeated startup waiting while
keeping the same model, RNG and private worker caches. See
[actual inference evidence](../docs/validation/wn2-a6000-final-inference.json).
It keeps the packaged BF16 model and deterministic GPU policy and generates
device-local autotuning data.
All workers claim unfinished members from a shared queue, so additional GPU
allocations can join later without changing member RNG identities. Worker
manifests record runtime/device information. Two-process queue checks passed
for both model runners; two host gate tests reject profiling, incomplete-stage
and modified packages. These checks do not establish GPU inference capacity.

`score_marginals.py --predictions DIR --data DATA --case 0 --pooling GEOMETRY
--output SCORES/case-000.json` scores one complete 56-member case, including
spherical mean/max panels. `compare.py --baseline BASELINE_SCORES --candidate
CANDIDATE_SCORES --output comparison.json` requires all eight cases and matching
truth, protocols, evaluator source and geometry identities.

Install `paper-comparison/requirements.lock.txt` into a separate operator
scoring environment and pass that interpreter with `--python` to the core
and pooling dispatchers. Dask is required to concatenate the 56 member stores
lazily; replacing lazy reads with eager reads would load the full ensemble.
Both dispatchers check the backend before submitting work, and scorers bound
Dask workers to the allocated CPU count. `dispatch_scores.py --source` selects
the frozen scorer snapshot without changing a snapshot already used by a job.

`dispatch_pooling.py` independently submits spherical mean/max scoring every
30 minutes once all 56 members of a case exist. It uses `--pooling-only` to
avoid repeating native-grid metrics, with separate `pooling-scores/MODEL`
outputs. Pair these directories using the same `compare.py`; all eight dates
and identical geometry/truth provenance remain required. CPU jobs request
8 cores, 64 GiB and a 12-hour limit; full-case runtime remains unmeasured.

The required remaining endpoints are tracked in `supplemental-endpoints.json`.
`dispatch_scores.py` runs every 30 minutes on the operator host and submits CPU
marginal scoring when a case has all 56 completed members. It records submission
receipts, avoids duplicate submissions and leaves failed jobs for investigation.
It does not send chat notifications or claim that a queued score has passed.

The baseline runner reuses the deterministic spherical grid via the upstream
`sample(..., grid=...)` argument. The original path reconstructed a roughly
5.6 GiB Legendre basis for each variable/noise call, exhausting host memory
before inference. `check_native_noise_cache.py` verifies exact small-grid noise
parity across two variables and two RNG seeds in the pinned model image.
The sampler and spectrum remain unchanged. Full-resolution GPU forecasts are
now running; completed 15-day ensembles and scores remain pending. Original
failed-start artifacts remain outside the repository.

Full GenCast stays in float32. Official GPU guidance requires approximately
60 GB VRAM and 300 GB host RAM during compilation; 48 GB A6000 allocations
were withdrawn after failed initialization. Current jobs target A100 80 GB and
RTX PRO 6000 96 GB cards. The launcher staggers first compilation within each
node and then leaves initialized GPUs forecasting concurrently.

`prepare_rev.py` prepares 1993–2016 ERA5 thresholds by month and hour. `rev.py`
implements potential REV by aggregating confusion matrices before maximizing
over all ensemble decision thresholds. Pass `--rev-climatology DIR` to the
scorer; the automatic dispatcher includes this argument and waits for verified
thresholds before submitting a case.

REV threshold preparation has passed verification for all 48 month/hour groups,
including file hashes, 0.25-degree grid, finite values and quantile monotonicity.
Its potential-REV implementation agrees with pinned WeatherBench-X to 2.6e-14
on independent random panels. Real forecast scores are still pending.

The current baseline has completed its first 12/24/36-hour predictions. Observed
steady A100 time is approximately 119 seconds per 12-hour step, implying about
56 hours for all 448 trajectories using only eight A100s, before other overheads.
Additional 96 GB GPU jobs are queued; their availability cannot be assumed.

## Cyclone verification

`tracker-source.lock.json` pins the exact TempestExtremes 2.1 archive.
`tracker-protocol.json` records detection and stitching commands, including the
units and sign corrections justified by the cited executable methods. The
operator built DetectNodes and StitchNodes against classic NetCDF and verified
the first real HRES case: five tracks with 73 points. These are tracker outputs,
not named-storm matches or model evaluation scores.

`prepare_tracker.py` prepares shared ten-day HRES history plus fifteen-day HRES
verification trajectories. All eight inputs have verified receipts.
`export_tracker_forecast.py` replaces only the future with a completed, verified
forecast member; its end-to-end forecast run awaits a complete member. Run
`run_tracker.py` on each exported trajectory. Successful execution steps have
checksummed receipts so validation can resume without repeating detection.
The runner also checks logs because this tracker version can return exit code
zero after a C++ exception. `tracker-truth.sbatch` runs the remaining seven HRES
cases on CPUs; all eight truth runs have now passed output verification.

`prepare_named_cyclones.py` selects main named tracks from the frozen IBTrACS
snapshot using its ISO timestamps. Numeric CF timestamps have sub-millisecond
decoding noise; both time representations must agree within one millisecond.
No track interpolation is used. `cyclone_metrics.py` implements initial matching,
spherical ensemble means, common storm/lead position comparisons and one-degree
cell-event confusion matrices. Six tests cover dateline wrapping, unique
initial matching, member dropout, paired support, cell counting and timestamp
decoding. Full forecast scoring remains pending. `cyclone-score-protocol.json`
records the numerical conventions not fully specified by the paper's prose.

`score_cyclones.py` verifies all 56 tracker receipts, their bindings to prediction
manifests and shared history, and model/export/executable consistency before
producing a case's position rows and cell-event confusion matrices.
`compare_cyclones.py` requires all eight dates, computes errors on common
storm/lead pairs and aggregates confusion matrices before computing potential
REV. Empty position support is reported as missing, never as zero error.
`dispatch_cyclones.py` runs every 30 minutes from a frozen operator source copy:
it submits each completed forecast member for CPU export/tracking, then submits
case scoring after every member is tracked. Failed jobs require investigation
and are not automatically resubmitted.

Cyclone tracking currently uses common 12-hour frames for both models. The paper
reports a 12-hour FGN comparison as well as its main native 6-hour FGN tracking
result; this workflow must be labeled as the common-12-hour comparison.

## Spherical spectra

`spherical_spectra.py` uses the pinned Dinosaur 1.2.1 real spherical harmonic
basis and equiangular-with-poles quadrature on the native 721 x 1440 grid.
Float64 NumPy contractions avoid compiling the large basis into XLA. Power is
the sum of squared orthonormal coefficients over zonal orders, divided by
4*pi, retaining total wavenumbers 0 through 719. No latitude regridding or planar
FFT is used. The backend and normalization are explicit numerical conventions;
the paper does not specify all such implementation details.

`check_spherical_spectra.py` passed analytic constant, dipole and degree-two
mode checks, native-transform parity, Parseval on those modes and longitude
rotation invariance in the pinned model image. A complete real HRES date
(300 panels, 720 wavenumbers each) took 88 seconds, with approximately 6 GiB peak
resident memory. These are transform/reference validations, not model scores.

`prepare_truth_spectra.py` generates all ten requested fields at all thirty
leads. `score_spectra.py` verifies every member before separately computing the
mean of individual-member powers and the power of the ensemble-mean field.
`compare_spectra.py` requires all eight date reports and matched truth and
transform provenance. Two aggregation tests verify power averaging, rejection
of changed truth and symmetric handling of missing truth support.

Five z850 fields across three dates each contain 4-6 missing HRES grid points;
an independent upstream read confirmed missing values at the first affected
field's locations. A full-sphere spectrum is undefined for an incomplete field.
The explicit policy in `spectra-protocol.json` retains a missing-truth record
and excludes that case/field/lead from both models' paired spectral average,
reporting the actual count and excluded dates. It does not interpolate or
zero-fill those points. All eight forecast dates and the other metric protocols
are retained. This limitation must accompany spectral reports.

`dispatch_spectra.py` runs every 30 minutes from a frozen source copy and submits
CPU jobs when a case has all 56 completed members and its reference spectra.
Actual complete model spectral scores remain pending.

All eight HRES reference reports have now passed inventory/provenance checks:
2,400 requested panels, 2,395 complete spectra and five explicitly missing
truth spectra. Successful and missing panels both remain in the case reports.

## Precipitation SEEPS

`seeps-protocol.json` follows the monthly 1980-2015 climatology described in
the GraphCast precipitation methods cited by GenCast Appendix F.4. The source
PDF and its checksum are pinned in `seeps-climatology-source.lock.json`.
The WeatherBench default 1990-2019 sliding-window climatology is a different
method and is not used. Monthly samples pool all four six-hourly endpoints.
Separate statistics are computed for 12/24-hour accumulation and 0.1/0.25 mm
dry thresholds. Wet thresholds are linear two-thirds quantiles of non-dry
samples; the scoring matrix uses the corresponding monthly dry fraction.

The paper's strict dry boundary (`precipitation < threshold`) is retained;
WeatherBench-X uses `<=` at that edge and an annual-mean dry fraction. The matrix
implementation agrees with pinned WeatherBench-X away from the differing dry
boundary. Six SEEPS tests cover its nine category combinations, wet/dry masks,
empty support, the explicit boundary, accumulation context, quantiles and paired
aggregation. Together with the eight existing subset tests, 14 tests passed.

`stage_seeps_rain.py` caches the original compressed ERA5 chunks, including
18 hours before each year for accumulation continuity. All 36 required years
are staged; the first cached year's full-grid field and time coverage were
independently checked against upstream. `prepare_seeps.py` hashes consumed
chunks and produces monthly statistics. All twelve months and all 48
accumulation/dry-threshold combinations have passed verification. `verify_seeps.py` checks
all 48 combinations, sample counts, sources, finite eligible thresholds and
monotonicity before publishing the readiness receipt.

Pass `--seeps-climatology DIR` to `score_marginals.py` to score ensemble-mean
12/24-hour rain; 24-hour scores begin at lead 24. The dispatcher waits for full
climatology verification and supplies this argument. `compare.py` reports SEEPS
and 1-SEEPS with paired differences and valid area, rejecting changed support.
Complete real ensemble SEEPS scores remain pending.

## Automatic paired aggregation

`dispatch_comparisons.py` runs every 30 minutes after per-case scoring. For
each of native metrics, spherical pooling, spectra and cyclones, it requires
all eight verified date reports for both models before submitting aggregation.
It produces each WN2 seed versus GenCast and seed 1 versus seed 0 under
`comparisons/`, retaining source score hashes. Seed differences are descriptive;
two runs and eight dates do not establish statistical significance. Each
endpoint completes independently; none alone establishes full track acceptance.

## Simple references added by this track

`persistence-dry` supplements GenCast with a no-training reference. It holds
**every instantaneous weather variable** at its initialization value, at every
forecast lead. Wind speed and geopotential differences are derived from those
held fields. Precipitation is **zero for each 12-hour interval**, and hence zero
for 24-hour accumulations; this is a dry forecast, not persistence of an
accumulated quantity. These references are our additions, not a claimed
reproduction of a persistence scorecard from the FGN paper.

The reference uses the same dates, truth masks, area weights, native-grid
CRPS/RMSE, derived fields, precipitation SEEPS, extreme-event REV and spherical
average/max pooling as the model runs. It is a point-mass forecast: spread is
zero, CRPS reduces to absolute error, and it provides no calibrated ensemble
uncertainty. Two identical copies suffice for these computations, exactly
representing 56 identical members without storing repeated forecast arrays.
Comparisons report both CRPS and RMSE skill (`1 - model / reference`), preserve
negative skill, and require all fixed dates. Zero reference error gives undefined
relative skill, rather than infinity. GPU cost is zero; CPU scoring is bounded.

REV already includes a historical-climate reference (REV=0). This is distinct
from a standalone seasonal-climatology forecast. The old seasonal Gaussian
reference has not been promoted to the current comparison: its artificial
spatial/member dependence should not be presented as a general baseline for
joint-distribution metrics. Power spectra remain comparisons to truth;
cyclone tracks require a separately specified track reference and matching
population, so neither is silently assigned a temperature-style baseline.

Run one case with the same frozen source snapshot as model scoring:

```bash
python paper-comparison/score_marginals.py --baseline persistence-dry \
  --data ROOT/data --case 0 --rev-climatology ROOT/rev-climatology \
  --seeps-climatology ROOT/seeps/climatology \
  --output ROOT/scores/persistence-dry/case-000.json
```

Run `dispatch_simple_baselines.py --root ROOT --source SNAPSHOT --python PYTHON
--max-active 2 --submit` alongside the model dispatchers. It schedules all core
cases before pooling, then automatically compares complete reference/model date
sets. Receipts prevent duplicate submission and unbounded retries. When adding
references to archived model scores, `--reference-scorer OLD/score_marginals.py`
records that scorer; comparisons still require identical metric implementations,
truth, protocols and settings. Existing model forecasts need not be rerun.

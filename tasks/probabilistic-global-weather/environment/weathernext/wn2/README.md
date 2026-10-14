# WN2 implementation and operator handbook

The canonical contract is [environment/contract.md](../environment/contract.md).
The build selects model code for the [agent starter](../environment/baseline/starter/README.md);
profiling, acquisition and multi-run orchestration remain operator tools.
See [READINESS.md](../docs/READINESS.md) for acceptance status and
[the result summary](../docs/validation/results-summary/README.md) for measured results.

## Data and protocol

`training-protocol.json` freezes 32 training windows from 2017–2020. Statistics
and cyclone wind calibration use training data only. Visible development uses
four 2021 dates. Formal evaluation uses `protocol.json`: eight preselected 2022
dates, 56 members, the native 0.25° grid and 12-hour leads through day 15.
`precipitation-protocol.json` defines 12-hour ERA5 accumulation; inference sums
two native six-hour predictions. Paper-comparison protocol paths are aliases
of these same files.

Use [the data preparation instructions](../environment/data/MATERIALIZE.md) to acquire
and verify training, development and the complete formal benchmark. The formal
bundle includes context, weather and precipitation truth, climatology, spherical
pooling neighborhoods, spectra and cyclone references. It is mounted only in
the verifier. Our eight-date 2022 subset does not reproduce the paper's full-year
2023 score table.

## Training and packaging

`train_stage.py` implements all eleven native curriculum stages, retaining Adam
moments across compatible transitions with effective batch 64 on four GPUs.
The default agent recipe is the fixed 580-update schedule in
`environment/starter/schedule.json`, with 24 hours for training and 30 minutes
for packaging. The original 562-update recipe is retained as a lower-compute
comparison. `plan_schedule.py` and `collect_profiles.py` are operator tools for
the original common-factor timing method; they do not generate the default
extended schedule.

The operator `train.py` and `slurm/wn2_curriculum.sbatch` execute the frozen
curriculum and export `submission-cold`. Their success records training and
packaging only. Submit evaluation separately so CPU scoring does not hold GPUs.
The delivered starter has its own single-run `train.py`; it does not invoke HPC
orchestration. `package_model.py` exports code, final weights and training-only
statistics, optionally with a compatible compilation cache.

Training uses `/opt/weather-python/bin/python`; CPU scoring uses
`/opt/wn2-eval/bin/python` with `PYTHONPATH=/opt/weatherbenchX`. Keep these pinned
environments separate. Training enforces deterministic GPU operations, fixed
cuBLAS workspace configuration and recorded autotuning choices.

## Independent evaluation

Follow [the acceptance commands](../slurm/wn2_acceptance.md). `verify_case.py`
relocates the submission and runs its actual `infer.sh` inside the offline
namespace boundary. The interface writes one consolidated Zarr store per
requested member and lead window. Predictions must be finite and match all requested
coordinates, fields and accumulation semantics.

`verify_track.py` defaults to the [windowed pipeline](WINDOWED_EVALUATION.md)
with a 2 TB allowance in `test.sh` (eight concurrent cases on eight GPUs; 1 TB fits four) and lossless
v3 continuation. Repackage existing stock
weights with `upgrade_inference.py`; no training rerun is needed.

Legacy split execution requires `--legacy-case-mode` in both phases:
`--phase forecast --keep-predictions` generates retained forecasts (8 TiB), then
`--phase score` validates receipts and runs the CPU endpoint suite.
Both phases must pass before a numeric reward is accepted. Weather CRPS
is normalized with training-only scales; precipitation, derived fields, pooling,
REV, SEEPS, spectra and cyclone diagnostics remain separately interpretable.
The [paper-comparison tools](../paper-comparison/README.md) also evaluate GenCast
and simple references against the same truth. Baseline weights never enter
candidate training.

The visible development helper is `development.py`; its command and smaller
four-date scope are in [evaluation/README.md](../environment/evaluation/README.md).
`evaluate.py`, `merge_scores.py` and their low-level helpers support development
and reward calculation. `compare_legacy_reports.py` reads archived reduced-panel
reports only and is not a formal acceptance entry point.

Actual GPU rollout, container execution and complete scientific scoring are
separate acceptance evidence; passing unit tests alone does not establish them.

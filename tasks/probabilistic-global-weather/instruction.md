# Train and improve a native WN2 ensemble

## Task and goal

Develop one portable WeatherNext 2 model that forecasts global weather from two six-hourly context
frames. It delivers 56 ensemble members at 12-hour leads through 15 days on the native 721×1440
grid. Probabilistic medium-range forecasts with calibrated uncertainty are the basis of early
warnings for extreme weather.

The reward is `1/(1 + normalized CRPS)`: empirical ensemble CRPS for ten weather variables,
including all 13 pressure levels of the six atmospheric fields, each divided by its frozen
training-only HRES standard deviation, averaged equally over levels and leads within a variable and
then over variables. Larger is better. Precipitation has a separate ERA5-verified score; derived
fields, spatial pooling, precipitation SEEPS, extreme-event value, spectra and cyclone tracks are
reported separately. The task author's 580-update starter reached reward 0.680651 (normalized CRPS
0.469183) on the eight-date 2022 test panel. Aim above it.

## Workspace

- `/workspace/contract.md`: the `infer.sh` interface (v3 continuation), the output stores, inference
  limits and the scoring.
- `/workspace/data/` (read-only):
  - `README.md` describes the data;
  - `training/`: the fixed 2017–2020 training windows and the training-only normalizers and
    cyclone labels;
  - `development/`: four visible 2021 dates (context frames, weather and precipitation truth) for
    model selection.
- `/workspace/baseline/starter/`: the verified native WN2 training workflow (`train.py`,
  `schedule.json` with the eleven-stage, 580-update curriculum) with its results.
- `/workspace/references/`: the model family, the metric implementations and related work.
- `/workspace/evaluation/`: development evaluation on the four 2021 dates, with the verifier's
  scoring code.
- `/workspace/research/`: writable space for runs, checkpoints and forecasts.

## Rules

- Initialize all 183,785,575 parameters randomly; no pretrained weights. Retain the native
  two-sample CRPS training objective and the effective training batch of 64. Preserve weights and
  optimizer moments at stage transitions; parameter time-coordinate names change between the
  initial 12-hour and six-hour stages.
- Train only on the fixed 2017–2020 windows with the supplied training-only normalizers and cyclone
  labels. Use the 2021 development dates for model selection only. The eight-date 2022 test panel
  is not training or model-selection data.
- Do not fetch test truth, cache future weather, encode date-specific answers or call a forecast
  service. Calendar forcings may be computed from the supplied timestamps. The retrospective public
  sources are not claimed to be secret; independent inference receives neither training files nor
  scoring truth.
- There is no network.

## Submission

Keep one self-contained package at `/workspace/submission`, from early on. Its `infer.sh` must work
after relocation, on unseen dates and arbitrary case IDs and order, using one GPU per case, and
implement the v3 continuation interface in the contract. The verifier runs it and computes the
reward; submitted scores or prediction-only archives are not accepted. The package may be read at
any moment, so replace it as a whole: the starter packages into a staging directory before
exporting. Keep training runs and checkpoints under `/workspace/research`.

## Continuous research

The starter has already been trained in this setup (13.1 h on four H100s) and scored:
`baseline/starter/RESULTS.md` has its reward, `reproduction/` its per-stage training reports, and
`reproduction/test-evaluation/` its CRPS, RMSE and spread by variable, level and lead next to GenCast and
persistence. Its score on the 2021 development dates was not recorded. Do not retrain or re-evaluate the
starter to obtain it: use that breakdown to find where the starter is weak, compare your own candidates
on the 2021 dates, check your pipeline with a short run and publish a package early. Then keep improving: evaluate on the 2021 dates (`/workspace/evaluation/README.md`,
a new output directory each time), analyze errors by variable, level and lead, form a hypothesis,
train and evaluate a candidate, and publish it when it is better. Then start the next experiment.
Do not stop to hand in a final answer; a published model or a reached score is a milestone, not the
end of the task.

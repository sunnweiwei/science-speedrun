# Probabilistic global weather — native WN2

## 1. Overview

Train all 183,785,575 WeatherNext 2 parameters from random initialization with a shortened
eleven-stage curriculum, deliver a portable model, and evaluate it independently on eight fixed 2022
dates, 56 ensemble members, 30 twelve-hour leads through 15 days and the native 0.25° global grid.

The single test protocol covers 82 weather field/level combinations, derived fields, 12/24-hour
precipitation, spherical average/max pooling, SEEPS, extreme-event relative economic value, spectra
and cyclone tracks. The reward is `1/(1 + normalized CRPS)`, training-scale-normalized empirical
weather CRPS; all auxiliary metrics are reported separately. This is a limited paper-method subset,
not a reproduction of the published 2023 score table.

- **Status:** training reproduced by the reviewers; their independent score is pending (section 6).
  The author's run scored reward **0.680651** (normalized CRPS 0.469183).

## 2. Data

Prepared resources, including a frozen IBTrACS snapshot, are a public Hugging Face dataset
([`sanxing/ai4sci-probabilistic-global-weather`](https://huggingface.co/datasets/sanxing/ai4sci-probabilistic-global-weather),
pinned commit, 206 GB, anonymous download, every shard and extracted file verified). Sources: ERA5
and ARCO-ERA5, WeatherBench HRES-fc0, NOAA IBTrACS; each keeps its own terms.
[`environment/data/MATERIALIZE.md`](environment/data/MATERIALIZE.md) has the download, scopes,
storage and the maintainer-only upstream rebuild:

```bash
python3 -m pip install 'huggingface_hub>=1.0,<2'
bash probabilistic-global-weather/environment/data/materialize.sh "$TASK_DATA_DIR"
```

```text
$TASK_DATA_DIR/
  training-2017-2020/{case-000 … case-031, statistics/}     agent: /workspace/data/training
  development-{inputs,weather,precipitation}-2021/          agent: /workspace/data/development/*
  evaluation-resources/                                     verifier only (/benchmark)
  IBTrACS.since1980.v04r01.nc
```

Training uses 32 frozen windows from 2017–2020; development uses four 2021 dates; the test uses
eight preselected 2022 dates.

## 3. Workspace

```text
/workspace/
  instruction.md, contract.md
  data/README.md, data/training/, data/development/{inputs,weather,precipitation}/   read-only mounts
  baseline/starter/     train.py, schedule.json (580 updates) and the WN2 runtime, README, RESULTS
  references/           model family, metric implementations
  evaluation/           development evaluation on the four 2021 dates (verifier scoring code)
  submission/           empty at start
  research/             empty, writable, large disk
```

The starter is assembled at image build time from the shared `environment/weathernext/wn2/` library, `environment/weathernext/slurm/` helpers and
`environment/baseline/starter/` by `environment/build_starter.py`, which strips profiling-only entry
points. [Training design](docs/WN2_SHORT_TRAIN.md).

## 4. Evaluation

The verifier (`tests/`) mounts only the dedicated `evaluation-resources` bundle. `verify_case.py`
copies the model to a fresh directory and runs it in an offline Bubblewrap namespace containing
only the runtime, the model, the context and the output; truth and training data never enter it,
and scoring starts after that namespace has exited. The default evaluator uses a 2 TB disk
allowance, scoring consecutive time windows and resuming from saved model state without
recomputing earlier forecasts ([windowed evaluation](environment/weathernext/wn2/WINDOWED_EVALUATION.md)); retained-output
or split-phase evaluation needs up to 8 TiB. It evaluates all eight cases at once, one per GPU, and
splits each window's scoring into one process per lead across the CPUs. With four cases at once
(1 TB, four H100s, 32 CPU cores) the full evaluation took 5.1 h in two waves; one wave of eight cases
on eight GPUs and 64 cores is expected to take about 2.5 h.

The weather reward divides each CRPS by its frozen training-only HRES standard deviation, averages
equally across levels and leads within each variable, then across the ten variables, and reports
`1/(1 + normalized CRPS)`. Precipitation (ERA5-verified), SEEPS, extremes, pooling, spectra and
cyclone tracks are separate panels. A failed prediction receives reward 0 with a failure record;
[scoring definitions and paper limitations](environment/weathernext/paper-comparison/README.md).

## 5. Running

```bash
export TASK_DATA_DIR=/large-disk/wn2-data TASK_PRIVATE_DIR=/large-disk/wn2-data   # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/weather-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p probabilistic-global-weather -a <agent> -m <model> --allow-agent-host <model API host>
```

**Resources (external budget, in `task.toml`).** Four GPUs (no specific model), 32 CPU cores,
384 GiB memory, 2 TiB storage for the agent; the reference recipe caps one training run at 24 hours
plus 30 minutes of packaging (measured 87.51 RTX PRO 6000 GPU-hours; 13.1 h on four H100s). The
separate full evaluation is limited to 12 hours on eight GPUs including CPU scoring (measured: 5.1 h
with four cases at a time on four H100s; about 2.5 h expected with all eight at once).

Both the development evaluator and the verifier run the model under Bubblewrap. On hosts that
restrict unprivileged user namespaces (Ubuntu 24.04 by default) Docker must allow it:
`tests/docker-compose.yaml` sets `seccomp`, `apparmor` and `systempaths` to `unconfined` and adds
`SYS_ADMIN`. The development compose file keeps only `seccomp:unconfined`; on such hosts the
development evaluator needs the same options (a decision for the operator, since they widen the
agent container's privileges). Standalone Docker commands without Harbor or Slurm:
[`docs/RUNNING.md`](docs/RUNNING.md).

## 6. Results and validation

| run | result |
| --- | --- |
| author, 580-update recipe, seed 0 (Apptainer runtime from the task Dockerfiles) | reward **0.680651** (normalized CRPS 0.469183); 2.20 % lower normalized CRPS than the original 562-update seed 0 |
| author, original 562-update seed 0 | reward 0.675805 (normalized CRPS 0.479717) |
| reviewers, same 580-update recipe retrained from scratch, Docker on four H100 80 GB | training: all eleven stages passed in 13.1 h, 183.8M-parameter package; **reward 0.674377** (normalized CRPS 0.482850), full verifier in Docker, 5.1 h |

The reviewers' score is an independent retraining, not a replay of the author's checkpoint: a fresh
uninterrupted 580-update run on different GPUs, where the author's extended seed 0 continued the
562-update run from update 556. It lands 0.2 % below the author's original seed and 0.9 % below the
extended one in reward, i.e. at the spread between the author's own two runs. Per variable it beats
the author's original seed on six of ten variables and trails on four, by at most 0.032 normalized
CRPS (10 m u-wind); geopotential agrees to 0.0001. The recipe's 2.20 % gain over the original seed is
therefore of the same size as run-to-run variation. The run went through the task's own
`tests/docker-compose.yaml` and `tests/test.sh`, all 8 dates, 56 members and every endpoint
(scores, pooling, spectra, cyclones); the record is
[parallel-evaluation-20260930.json](docs/validation/parallel-evaluation-20260930.json).

Two Docker-only faults were fixed on the way. Bubblewrap needs four container options on hosts that
restrict unprivileged user namespaces (in `tests/docker-compose.yaml`), and with GPUs attached the
NVIDIA runtime's `/proc/driver/nvidia/gpus/*` mounts block the namespace's own procfs, so
`environment/weathernext/wn2/test.sh` detaches them; either fault alone scores every case 0. The [evaluation report](docs/validation/results-summary/README.md) compares the
author's run with the original seed, GenCast and simple references;
[`docs/READINESS.md`](docs/READINESS.md) records the author's acceptance.

## 7. Limitations

- One model seed and a small seasonally selected training slice are deliberate reductions; this is
  not the published WeatherNext 2 training.
- The task fixes the method (random initialization of the native model, the two-sample CRPS
  objective, effective batch 64, the reference curriculum in the contract), which leaves less room
  for open-ended improvement than the other tasks.
- Full Docker/Compose execution of the verifier has not completed on a reviewer host yet (section 6).
- The retrospective public sources are not secret; the isolation is an execution boundary.

## 8. Citation and licenses

Task adaptation: Sanxing Chen (sc829). WeatherNext code (Apache-2.0, `environment/weathernext/wn2/LICENSE.weathernext-code`),
WeatherBench-X, ERA5/ARCO-ERA5 (Copernicus terms), WeatherBench HRES-fc0 and NOAA IBTrACS; preserve
the upstream code notice and each data source's attribution and terms.

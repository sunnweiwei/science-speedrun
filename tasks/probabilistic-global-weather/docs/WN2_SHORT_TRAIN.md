# Native WN2 training recipe

Train all 183,785,575 native parameters from random initialization with the
[single-run starter](../environment/baseline/starter/README.md), retaining parameters and
Adam state across eleven stages. Use four RTX PRO 6000 96-GB GPUs, native
two-sample CRPS, BF16 activations, FP32 parameters/loss reductions and effective
batch 64. Architecture, training windows and training-only normalization are
identical in the original and extended recipes.

## Stage schedule

| Stage | Source / grid / cadence | Original updates | Main recipe updates |
|---|---|---:|---:|
| Initial pretraining | ERA5 / 1° / 12 h | 410 | 410 |
| Cadence adaptation | ERA5 / 1° / 6 h | 102 | 102 |
| Resolution adaptation | ERA5 / 0.25° / 6 h | 32 | 32 |
| AR1 | HRES / 0.25° / 6 h | 8 | 8 |
| AR2 | HRES / 0.25° / 6 h | 4 | 4 |
| AR3–AR8 | HRES / 0.25° / 6 h | 1 each | 4 each |
| **Total** | | **562** | **580** |

The [original schedule](validation/training-curves/original-schedule.json) is
retained for reproducing the lower-compute reference. The stage definitions
and learning-rate peaks are unchanged. Warmup and cosine
schedules are sampled at compressed update midpoints; AR3–AR8 each use a
0.4-update warmup equivalent. Parameters and optimizer state are retained at
stage transitions. The final reference checkpoint was produced by continuing
from the original AR2 state at update 556; the main starter expresses the same
prefix and extended suffix as one from-scratch recipe.

## Resource budget

Training permits **24 hours × 4 GPUs = 96 GPU-hours**, plus 30 minutes for
packaging. The agent timeout was 24.5 hours when this recipe was designed (the task now uses the standard 8× H100, 48 h). The composed measured training
estimate is **87.51 RTX PRO 6000 GPU-hours**, or **21.88 hours on four GPUs**:
41.14 GPU-hours through AR2 and 46.37 GPU-hours for the extended suffix.
This is a sum of measured segments, not a measured uninterrupted 580-update run.
Original seed-0 training used 59.45 GPU-hours; the extended recipe adds about 47%.

Data preparation precedes training. Inference and CPU scoring have a separate
12-hour verifier envelope and are excluded from training costs. The operator
can split GPU inference, CPU output validation and CPU scoring to release GPUs
as soon as model execution ends. A resume retains the original start time;
it does not reset the training or packaging budget.

## Data and evaluation

Training uses 32 frozen windows from 2017–2020; development uses 2021. Evaluation
uses eight fixed 2022 dates, 56 members and twelve-hour leads through day 15 on
the native 0.25° global grid. Normalization and climatological resources follow
the declared training-only protocols. See the [submission contract](../environment/contract.md)
and [data preparation instructions](../environment/data/MATERIALIZE.md).

The main recipe reduces normalized test CRPS from **0.479717 to 0.469183**
(2.20%) versus the original checkpoint. Improvements occur in all ten native
variable aggregates and in 2,827/2,969 native/derived CRPS panels. These panels
are correlated, and cyclone results are mixed. The larger training budget is
part of the comparison; this is not an equal-compute ablation. GenCast remains
the stronger reference. Full plots, numeric tables, physical-unit errors and
limitations are in the [evaluation report](validation/results-summary/README.md).

Development evidence motivated the extended candidate; the 2022 test subset
also informed its promotion to the main recipe. These dates are therefore not
an untouched confirmation set for further recipe selection. The study does not
reproduce the paper's full-year 2023 experiment or establish statistical significance.

## Model and runtime

Use WeatherNext revision `89c4b2a77a1c57b328b909c575550fd2e5aadc9c` and the full
released configuration: 24 Transformer layers, width 768, six heads and mesh
refinement six. No pretrained parameters are used. The submitted artifact
contains final weights, training-only statistics and portable inference code.
Actual Docker/Compose execution remains unverified; validation uses the
Dockerfile-derived Apptainer runtime under the documented best-effort limitation.

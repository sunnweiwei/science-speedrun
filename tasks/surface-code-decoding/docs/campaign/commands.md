# Reproduction commands

Run from the `surface-code-decoding` directory. Data and outputs are intentionally
outside the checkout. Commands below use operator-chosen absolute directories;
the recorded campaign used CS Slurm and Python 3.11.14.

## Environment and public data

```bash
uv sync --frozen --python 3.11
export QEC_TASK_ROOT="$PWD"
export QEC_ASSET_ROOT=/absolute/path/to/persistent/assets
mkdir -p "$QEC_ASSET_ROOT/data" "$QEC_ASSET_ROOT/logs"
curl -fL --retry 3 \
  https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content \
  -o "$QEC_ASSET_ROOT/data/sycamore.zip"
```

Verify 315,490,804 bytes and MD5 `a7fd8b481c3087090093106382dc217d` before
extracting into `$QEC_ASSET_ROOT/data/sycamore`. The directory directly contains
the `surface_code_b...` condition directories. Retain the release's README and
CC-BY-4.0 attribution. Download size is 315 MB; expanded release is 548 MB.
Fixed training caches are additional, and are not required to run classical
baselines or packaged inference.

## Classical replay and simulation

```bash
sbatch --output="$QEC_ASSET_ROOT/logs/classical-%j.log" slurm/classical.sbatch
sbatch --output="$QEC_ASSET_ROOT/logs/prepare-%A-%a.log" slurm/prepare.sbatch
```

The first command locks one calibration choice per matching algorithm on the
validation labels, then decodes every odd test shot and rescores the release's
historical predictions. It saves counts and paired error arrays. The second
generates 20M training shots per d3 group and 60M per d5 group, plus independent
synthetic validation. New caches record circuit and payload hashes in schema v2.

## Native model and diagnostics

```bash
QEC_DATA_ROOT="$QEC_ASSET_ROOT/data/sycamore" \
  TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 \
  .venv/bin/python -m pytest tests -q
sbatch --output="$QEC_ASSET_ROOT/logs/overfit-%j.log" slurm/overfit.sbatch
sbatch --gres=gpu:v100:1 \
  --output="$QEC_ASSET_ROOT/logs/pilot-%A-%a.log" slurm/pilot.sbatch
```

The overfit and 1,000-step pilots are diagnostics, not benchmark results. Check
their recorded status, finite losses and learning gate before long training.
The native pilot used V100 because A5000 was queued; both cards passed the
TensorFlow 2.15 runtime checks. See `pilot-summary.json` for measured throughput.

## Bounded training and independent replay

`slurm/train.sbatch` trains one group per array element, defaults to a cumulative
24-hour stage budget, and resumes the exact cached-data cursor from `last.keras`.
`QEC_TRAIN_HOURS` overrides the cumulative budget; `QEC_CAMPAIGN` selects a new
artifact subdirectory. `--time` must include loading/checkpoint overhead and
must be considered separately from the trainer's cumulative cap.

`slurm/finetune.sbatch` uses only even hardware training/validation shots, with
an additional two-hour cap and at most 100 epochs. It retains pretrained Adam
state, lowers the learning rate to `1e-4`, and selects its own validation-best
checkpoint. Report this hardware adaptation separately from pretraining.

The campaign dependency chain is:

```text
representative training -> learning gate -> all pretraining arrays
    -> finetune array -> freeze -> replay array -> score
    -> terminal status (afterany)
```

Use Slurm `afterok` dependencies and `--kill-on-invalid-dep=yes`, and audit failed
or dependency-blocked jobs. The final status report uses `afterany` so failures
also produce an accounting/report snapshot. The learning gate requires a
completed full epoch and finite validation improvement, and makes no claim
about convergence or final experimental performance.
`freeze.sbatch` copies selected models and minimal inference code to immutable
packages, relocates them, creates fresh pinned environments, and constructs
detector-only inference inputs. `replay.sbatch` loads those relocated packages
and predicts all 3.25M odd test shots per stage. `score.sbatch` alone reads the
ground truth and produces both stage summaries and paired comparisons.

Keep the total allocated GPU budget, including retries and replay, below the
384-hour ceiling. Separate arrays do not share a concurrency throttle: their
simultaneous limits must sum to at most eight. Queue time consumes no GPU-hours.

## Interpretation

The primary metric fits log fidelity versus rounds 3–25 with a free intercept,
separately per patch/basis, then averages group LERs within distance. Bootstrap
intervals assume independent shots and conditions; they omit device drift,
fit misspecification and training-seed variation. Extrapolated F0 can slightly
exceed one, and is reported with diagnostics rather than clipped.

The odd-shot test is publicly accessible. The split prevents using its labels
for fitting or selection in this reproduction; it is not a private blind test.
The inference data interface excludes labels, but shared-host execution is not
a security boundary. Neither offline throughput nor per-shot batching timings
establish streaming or real-time quantum-control latency.

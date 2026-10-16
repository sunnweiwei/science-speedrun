#!/bin/bash
#SBATCH --job-name=olmo-few-shot-probes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
# CPU-only probe fitting: no GPU resource is requested or used.
OLMO_DATA=${OLMO_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth}
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/docs/analysis/few_shot_awf.py" \
  --data-root "$OLMO_DATA" \
  --names large-random-eval-12536 large-raw large-official-reference \
    large-run001-step500 large-run001-step2000 large-run002-step500 \
    large-run002-step2000 large-run003-step1000 \
  --output "$OLMO_DATA/results/large-few-shot-development.json"

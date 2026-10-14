#!/usr/bin/env bash
#SBATCH --job-name=wn2-training-data
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_SNAPSHOT=${WN2_ROOT}/training-data-runners/${SLURM_JOB_ID}
mkdir -p "${WN2_SNAPSHOT}"
cp "${TASK_ROOT}"/wn2/*.py "${TASK_ROOT}/wn2/training-protocol.json" "${WN2_SNAPSHOT}/"
export PYTHONPATH=${WN2_ROOT}/weatherbenchX
export OPENBLAS_NUM_THREADS=2
exec "${WN2_ROOT}/eval-py312/bin/python" "${WN2_SNAPSHOT}/materialize_training.py" \
  --protocol "${WN2_SNAPSHOT}/training-protocol.json" \
  --output "${WN2_ROOT}/training-2017-2020" --case-index "${SLURM_ARRAY_TASK_ID:-0}"

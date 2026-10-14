#!/usr/bin/env bash
#SBATCH --job-name=wn2-inputs
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
export PYTHONPATH=${WEATHER_WBX_SOURCE:-${WN2_ROOT}/weatherbenchX}
export OPENBLAS_NUM_THREADS=2
WN2_SNAPSHOT=${WN2_ROOT}/truth-runners/${SLURM_JOB_ID}
mkdir -p "${WN2_SNAPSHOT}"
cp "${TASK_ROOT}"/wn2/*.py "${WN2_SNAPSHOT}/"
cp "${WEATHER_WN2_PROTOCOL:-${TASK_ROOT}/wn2/protocol.json}" "${WN2_SNAPSHOT}/protocol.json"
exec "${WEATHER_EVAL_PYTHON:-${WN2_ROOT}/eval-py312/bin/python}" \
  "${WN2_SNAPSHOT}/materialize_inputs.py" --protocol "${WN2_SNAPSHOT}/protocol.json" \
  --output "${WEATHER_WN2_INPUTS:-${WN2_ROOT}/inputs-2022-draft}" --case-index "${SLURM_ARRAY_TASK_ID:-0}"

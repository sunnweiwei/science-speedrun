#!/usr/bin/env bash
#SBATCH --job-name=wn2-curriculum
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=18:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
export WEATHER_RUNTIME_ROOT=${WEATHER_RUNTIME_ROOT:-/usr/project/xtmp/sc829/weather-track/runtime-final}
export PYTHONPATH=${WN2_ROOT}/weatherbenchX
export OPENBLAS_NUM_THREADS=2
exec "${WN2_ROOT}/eval-py312/bin/python" "${TASK_ROOT}/wn2/train.py" \
  --data "${WN2_ROOT}/training-2017-2020" --plan "${WEATHER_WN2_PLAN:-${WN2_ROOT}/frozen-plan}" \
  --image "${WN2_IMAGE}" --eval-python "${WN2_ROOT}/eval-py312/bin/python" \
  --output "${WN2_ROOT}/curriculum-runs/${SLURM_JOB_ID}"

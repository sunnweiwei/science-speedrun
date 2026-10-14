#!/usr/bin/env bash
#SBATCH --job-name=wn2-reference-finalize
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_RUN=${WN2_ROOT}/reference-scores/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${TASK_ROOT}"/wn2/*.py "${TASK_ROOT}"/wn2/*protocol.json "${WN2_RUN}/code/"
export PYTHONPATH=${WN2_ROOT}/weatherbenchX
export OPENBLAS_NUM_THREADS=2
exec "${WN2_ROOT}/eval-py312/bin/python" "${WN2_RUN}/code/collect_reference.py" \
  --package "${WEATHER_EVAL_PACKAGE:-${WN2_ROOT}/persistence-package}" \
  --runs "${WN2_ROOT}/baseline-runs" --output "${WN2_RUN}/result" \
  --protocol "${WN2_RUN}/code/protocol.json" \
  --precipitation-protocol "${WN2_RUN}/code/precipitation-protocol.json" \
  --scales "${WN2_ROOT}/training-2017-2020/statistics/hres_025deg_6h/std.nc"

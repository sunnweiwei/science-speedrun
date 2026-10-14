#!/usr/bin/env bash
#SBATCH --job-name=wn2-training-stats
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
WN2_SNAPSHOT=${WN2_ROOT}/training-stat-runners/${SLURM_JOB_ID}
mkdir -p "${WN2_SNAPSHOT}"
cp "${TASK_ROOT}"/wn2/*.py "${TASK_ROOT}/wn2/training-protocol.json" "${WN2_SNAPSHOT}/"
WN2_ACTION=(--case-index "${SLURM_ARRAY_TASK_ID:-0}")
if [[ ${WEATHER_STATS_FINALIZE:-0} == 1 ]]; then WN2_ACTION=(--finalize); fi
exec apptainer exec --containall --cleanenv --env JAX_PLATFORMS=cpu \
  --env OPENBLAS_NUM_THREADS=2 --env OMP_NUM_THREADS=2 \
  --bind "${WN2_ROOT}:/data" --bind "${WN2_SNAPSHOT}:/code:ro" \
  "${WN2_IMAGE}" /opt/weather-python/bin/python /code/training_statistics.py \
  --root /data/training-2017-2020 --protocol /code/training-protocol.json \
  --output /data/training-2017-2020/statistics "${WN2_ACTION[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=wn2-training-labels
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
WN2_SNAPSHOT=${WN2_ROOT}/training-label-runners/${SLURM_JOB_ID}
mkdir -p "${WN2_SNAPSHOT}"
cp "${TASK_ROOT}/slurm/prepare_wn2_cyclones.py" "${TASK_ROOT}/wn2/training-protocol.json" "${WN2_SNAPSHOT}/"
mapfile -t WN2_TIMES < <("${WN2_ROOT}/eval-py312/bin/python" - "${WN2_SNAPSHOT}/training-protocol.json" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import json,sys,numpy as np
p=json.load(open(sys.argv[1]));t=np.datetime64(p['initializations'][int(sys.argv[2])])
for offset in p['offset_hours']:
    if offset>0:print(str(t+np.timedelta64(offset,'h')))
PY
)
WN2_CASE=$(printf 'case-%03d' "${SLURM_ARRAY_TASK_ID:-0}")
for WN2_RES in 1 .25; do
  apptainer exec --containall --cleanenv --env JAX_PLATFORMS=cpu \
    --env OPENBLAS_NUM_THREADS=2 --env OMP_NUM_THREADS=2 \
    --bind "${WN2_ROOT}:/data" --bind "${WN2_SNAPSHOT}:/code:ro" \
    "${WN2_IMAGE}" /opt/weather-python/bin/python /code/prepare_wn2_cyclones.py \
    --ibtracs /data/IBTrACS.since1980.v04r01.nc \
    --output "/data/training-2017-2020/${WN2_CASE}/cyclones-${WN2_RES}" \
    --resolution "${WN2_RES}" --times "${WN2_TIMES[@]}"
done

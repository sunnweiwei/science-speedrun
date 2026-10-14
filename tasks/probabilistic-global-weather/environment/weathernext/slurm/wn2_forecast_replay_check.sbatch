#!/usr/bin/env bash
#SBATCH --job-name=wn2-replay-hash
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:45:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
: "${WEATHER_REPLAY_RUN:?Set completed cold/warm run directory}"
WN2_CODE=${WN2_ROOT}/replay-checks/${SLURM_JOB_ID}
mkdir -p "${WN2_CODE}"
cp "${TASK_ROOT}/wn2/check_forecast_replay.py" "${TASK_ROOT}/wn2/training_data.py" "${WN2_CODE}/"
exec "${WN2_ROOT}/eval-py312/bin/python" "${WN2_CODE}/check_forecast_replay.py" --run "${WEATHER_REPLAY_RUN}"

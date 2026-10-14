#!/usr/bin/env bash
#SBATCH --job-name=weather-wn2-full
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=01:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
WN2_RUN=${WN2_ROOT}/runs/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}"
cp "${TASK_ROOT}/slurm/wn2_validation.py" "${WN2_RUN}/runner.py"
sha256sum "${WN2_IMAGE}" > "${WN2_RUN}/image.sha256"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "${WN2_RUN}/hardware.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=2 > "${WN2_RUN}/gpu-memory.csv" &
WN2_MONITOR=$!
trap 'kill "${WN2_MONITOR}" 2>/dev/null || true' EXIT
WN2_RESULT=0
read -r -a WN2_MODELS <<< "${WEATHER_WN2_MODELS:-wn2 cyclones}"
read -r -a WN2_MODES <<< "${WEATHER_WN2_MODES:-forward lora}"
for WN2_MODEL in "${WN2_MODELS[@]}"; do
  for WN2_MODE in "${WN2_MODES[@]}"; do
    apptainer exec --nv --containall --cleanenv \
      --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
      --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=8 \
      --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --env "XLA_FLAGS=${WEATHER_WN2_XLA_FLAGS:-}" \
      --bind "${WN2_ROOT}/assets:/assets:ro" --bind "${WN2_RUN}:/run" \
      "${WN2_IMAGE}" /opt/weather-python/bin/python /run/runner.py \
      --assets /assets --output "/run/${WN2_MODEL}-${WN2_MODE}" \
      --model "${WN2_MODEL}" --mode "${WN2_MODE}" --precision "${WEATHER_WN2_PRECISION:-float32}" \
      > "${WN2_RUN}/${WN2_MODEL}-${WN2_MODE}.log" 2>&1 || WN2_RESULT=1
  done
done
exit "${WN2_RESULT}"

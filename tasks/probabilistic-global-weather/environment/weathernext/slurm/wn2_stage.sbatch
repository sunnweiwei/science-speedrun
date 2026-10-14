#!/usr/bin/env bash
#SBATCH --job-name=wn2-stage
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=05:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_IMAGE=${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}
WN2_RUN=${WN2_ROOT}/stage-runs/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${TASK_ROOT}"/wn2/*.py "${TASK_ROOT}/wn2/training-protocol.json" "${WN2_RUN}/code/"
cp "${TASK_ROOT}/slurm/wn2_validation.py" "${TASK_ROOT}/slurm/wn2_distributed.py" "${WN2_RUN}/code/"
sha256sum "${WN2_IMAGE}" > "${WN2_RUN}/image.sha256"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=5 > "${WN2_RUN}/gpu-memory.csv" &
WN2_MONITOR=$!
trap 'kill "${WN2_MONITOR}" 2>/dev/null || true' EXIT
WN2_EXTRA=()
if [[ ${WEATHER_DEBUG_SINGLE_DEVICE:-0} == 1 ]]; then WN2_EXTRA=(--debug-single-device --debug-updates "${WEATHER_DEBUG_UPDATES:-1}"); fi
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=8 --env OPENBLAS_NUM_THREADS=8 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --env "XLA_FLAGS=${WEATHER_XLA_FLAGS:---xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_dump_autotune_results_to=/run/autotune.pb}" --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env JAX_COMPILATION_CACHE_DIR=/run/jax-cache --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  --env JAX_ENABLE_COMPILATION_CACHE=true --env JAX_RAISE_PERSISTENT_CACHE_ERRORS=true \
  --bind "${WN2_ROOT}/training-2017-2020:/data:ro" --bind "${WN2_RUN}:/run" \
  "${WN2_IMAGE}" /opt/weather-python/bin/python /run/code/train_stage.py \
  --data /data --statistics /data/statistics --protocol /run/code/training-protocol.json \
  --output /run/result --profile --stage-index "${WEATHER_STAGE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" "${WN2_EXTRA[@]}" \
  > "${WN2_RUN}/training.log" 2>&1

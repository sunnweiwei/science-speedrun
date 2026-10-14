#!/usr/bin/env bash
#SBATCH --job-name=wn2-cache-key
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_RUN=${WN2_ROOT}/cache-key-probes/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}"
cp "${TASK_ROOT}/slurm/check_wn2_cache_key.py" "${WN2_RUN}/"
for WN2_PHASE in cold warm; do
  apptainer exec --nv --containall --cleanenv \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 --env PYTHONHASHSEED=0 \
    --env XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env JAX_COMPILATION_CACHE_DIR=/run/jax-cache --env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    --env JAX_RAISE_PERSISTENT_CACHE_ERRORS=true --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env OPENBLAS_NUM_THREADS=2 --env PYTHONPATH=/model \
    --bind "${WN2_RUN}:/run" \
    --bind "${WN2_ROOT}/cached-rollout-runs/12533487/cold-package:/model:ro" \
    --bind "${WN2_ROOT}/development-inputs-2021:/data:ro" \
    "${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}" \
    /opt/weather-python/bin/python /run/check_wn2_cache_key.py --phase "${WN2_PHASE}" \
    > "${WN2_RUN}/${WN2_PHASE}.log" 2>&1
done

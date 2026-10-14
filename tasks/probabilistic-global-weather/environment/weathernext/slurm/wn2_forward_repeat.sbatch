#!/usr/bin/env bash
#SBATCH --job-name=wn2-forward-repeat
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_RUN=${WN2_ROOT}/forward-repeat-runs/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}"
cp "${TASK_ROOT}/slurm/check_wn2_forward_repeat.py" "${WN2_RUN}/"
WN2_XLA=--xla_gpu_deterministic_ops=true
if [[ ${SLURM_ARRAY_TASK_ID:-0} == 1 ]]; then WN2_XLA=--xla_gpu_exclude_nondeterministic_ops=true; fi
exec apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 \
  --env "XLA_FLAGS=${WN2_XLA}" --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env JAX_ENABLE_COMPILATION_CACHE=false --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env OPENBLAS_NUM_THREADS=2 --env PYTHONPATH=/model \
  --bind "${WN2_RUN}:/run" \
  --bind "${WEATHER_EVAL_PACKAGE:-${WN2_ROOT}/cached-rollout-runs/12533150/cold-package}:/model:ro" \
  --bind "${WN2_ROOT}/development-inputs-2021:/data:ro" \
  "${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}" \
  /opt/weather-python/bin/python /run/check_wn2_forward_repeat.py

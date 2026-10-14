#!/usr/bin/env bash
#SBATCH --job-name=wn2-query-oracle
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:10:00
set -euo pipefail
WN2_ROOT=/usr/project/xtmp/sc829/weather-track/wn2-pilot
WN2_RUN=${WN2_ROOT}/query-oracles/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_query_chunks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_boolean_masks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_packed_masks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_query_chunks.py" "${WN2_RUN}/code/"
WN2_EXTRA=()
if [[ ${WEATHER_ORACLE_MODE:-query} == boolean ]]; then WN2_EXTRA=(--boolean-mask); fi
if [[ ${WEATHER_ORACLE_MODE:-query} == packed ]]; then WN2_EXTRA=(--packed-mask); fi
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONHASHSEED=0 \
  --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --env XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env JAX_ENABLE_COMPILATION_CACHE=false \
  --bind "${WN2_RUN}:/run" \
  /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
  /opt/weather-python/bin/python /run/code/check_wn2_query_chunks.py \
  --require-gpu --output /run/result.json "${WN2_EXTRA[@]}" > "${WN2_RUN}/oracle.log" 2>&1

#!/usr/bin/env bash
#SBATCH --job-name=wn2-cudnn-oracle
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
WN2_RUN=${WN2_ROOT}/cudnn-oracles/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_query_chunks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_boolean_masks.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_query_chunks.py" "${WN2_RUN}/code/"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_cudnn_attention.py" \
  "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_cudnn_attention.py" "${WN2_RUN}/code/"
WN2_EXTRA=()
if [[ ${WEATHER_NATIVE_COARSE:-0} == 1 ]]; then WN2_EXTRA=(--native-coarse); fi
if [[ ${WEATHER_NATIVE_FINE:-0} == 1 ]]; then WN2_EXTRA=(--native-fine); fi
if [[ ${WEATHER_PMAP_REMAT:-0} == 1 ]]; then WN2_EXTRA+=(--pmap-remat); fi
if [[ ${WEATHER_SCAN_ENSEMBLE:-0} == 1 ]]; then WN2_EXTRA+=(--scan-ensemble); fi
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONHASHSEED=0 \
  --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --env "XLA_FLAGS=${WEATHER_CUDNN_XLA_FLAGS:---xla_gpu_exclude_nondeterministic_ops=true}" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env JAX_ENABLE_COMPILATION_CACHE=false \
  --bind "${WN2_RUN}:/run" \
  /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
  /opt/weather-python/bin/python /run/code/check_wn2_cudnn_attention.py \
  --output /run/result.json "${WN2_EXTRA[@]}" > "${WN2_RUN}/oracle.log" 2>&1

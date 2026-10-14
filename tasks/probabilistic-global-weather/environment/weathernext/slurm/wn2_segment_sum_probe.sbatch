#!/usr/bin/env bash
#SBATCH --job-name=wn2-segment-probe
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=00:15:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_RUN=${WN2_ROOT}/segment-probes/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}"
WN2_PROBE=${WEATHER_GRAPH_ORACLE:-check_wn2_segment_sum.py}
cp "${TASK_ROOT}/slurm/${WN2_PROBE}" "${TASK_ROOT}/slurm/wn2_validation.py" "${TASK_ROOT}/slurm/wn2_dense_chunks.py" "${WN2_RUN}/"
exec apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env PYTHONUNBUFFERED=1 \
  --env XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env OPENBLAS_NUM_THREADS=2 \
  --bind "${WN2_RUN}:/run" \
  "${WEATHER_IMAGE:-/usr/project/xtmp/sc829/weather-track/images-final/environment.sif}" \
  /opt/weather-python/bin/python "/run/${WN2_PROBE}"

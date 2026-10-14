#!/usr/bin/env bash
#SBATCH --job-name=wn2-image-gpu
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:06:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_ROOT=${WEATHER_WN2_ROOT:-/usr/project/xtmp/sc829/weather-track/wn2-pilot}
WN2_RUN=${WN2_ROOT}/harbor-gpu-boundary/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code/slurm" "${WN2_RUN}/code/wn2"
cp "${TASK_ROOT}/slurm/check_wn2_namespace.py" "${WN2_RUN}/code/slurm/"
cp "${TASK_ROOT}/wn2/namespace_inference.py" "${WN2_RUN}/code/wn2/"
apptainer exec --nv --containall --cleanenv --bind /dev/full:/dev/full \
  --bind "${WN2_RUN}:/results" --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --env WEATHER_MODEL_PYTHON=/opt/weather-python/bin/python --env OPENBLAS_NUM_THREADS=2 \
  "${WEATHER_HARBOR_IMAGE:-/usr/project/xtmp/sc829/weather-track/wn2-harbor-images-v3/tests.sif}" \
  unshare --user --map-root-user --mount bash -c \
  'mount --make-private / && exec /opt/wn2-eval/bin/python /results/code/slurm/check_wn2_namespace.py --gpu --output /results/boundary'

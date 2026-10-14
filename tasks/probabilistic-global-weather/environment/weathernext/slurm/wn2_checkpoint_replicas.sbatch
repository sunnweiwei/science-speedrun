#!/usr/bin/env bash
#SBATCH --job-name=wn2-checkpoint-replicas
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:a5000:2
#SBATCH --time=00:03:00
set -euo pipefail
WN2_RUN=/usr/project/xtmp/sc829/weather-track/wn2-pilot/checkpoint-replicas/${SLURM_JOB_ID}
mkdir -p "${WN2_RUN}/code"
cp "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/wn2_distributed.py" \
   "${SLURM_SUBMIT_DIR}/probabilistic-global-weather/slurm/check_wn2_checkpoint_replicas.py" "${WN2_RUN}/code/"
apptainer exec --nv --containall --cleanenv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false --bind "${WN2_RUN}:/run" \
  /usr/project/xtmp/sc829/weather-track/images-final/environment.sif \
  /opt/weather-python/bin/python /run/code/check_wn2_checkpoint_replicas.py --output /run/result.json

#!/usr/bin/env bash
#SBATCH --job-name=wn2-pipeline
#SBATCH --partition=compsci-gpu
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=36:00:00
set -euo pipefail
export WEATHER_PAIR_ROOT=${WEATHER_PAIR_ROOT:?}
export WEATHER_TRAIN_SEED=${WEATHER_TRAIN_SEED:?}
export WEATHER_RUNTIME_ROOT=/usr/project/xtmp/sc829/weather-track/runtime-final
export PYTHONPATH=/usr/project/xtmp/sc829/weather-track/wn2-pilot/weatherbenchX
export OPENBLAS_NUM_THREADS=2
exec /usr/project/xtmp/sc829/weather-track/wn2-pilot/eval-py312/bin/python "${WEATHER_PAIR_ROOT}/source/slurm/wn2_paired_pipeline.py"

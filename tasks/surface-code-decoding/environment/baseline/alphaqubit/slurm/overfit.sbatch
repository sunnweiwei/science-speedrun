#!/usr/bin/env bash
#SBATCH --job-name=qec-overfit
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.overfit \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --output "${QEC_ASSET_ROOT}/overfit-diagnostic.json" --max-steps 1000

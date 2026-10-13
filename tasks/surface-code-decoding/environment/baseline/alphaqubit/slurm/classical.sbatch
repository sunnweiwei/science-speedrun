#!/usr/bin/env bash
#SBATCH --job-name=qec-classical
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.classical \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --output "${QEC_ASSET_ROOT}/classical-v1"

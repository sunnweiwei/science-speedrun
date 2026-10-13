#!/usr/bin/env bash
#SBATCH --job-name=qec-learning-gate
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_CAMPAIGN:=native-v1}"
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.gate \
  --asset-root "${QEC_ASSET_ROOT}" --campaign "${QEC_CAMPAIGN}"

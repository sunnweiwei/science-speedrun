#!/usr/bin/env bash
#SBATCH --job-name=qec-data
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
if [[ "${qec_group}" == d3_* ]]; then qec_shots=20000000; else qec_shots=60000000; fi
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.cache \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --cache-root "${QEC_ASSET_ROOT}/synthetic" --group "${qec_group}" \
  --seed 0 --shots "${qec_shots}"

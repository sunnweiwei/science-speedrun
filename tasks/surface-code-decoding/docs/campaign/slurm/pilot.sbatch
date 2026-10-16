#!/usr/bin/env bash
#SBATCH --job-name=qec-pilot
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0,8%2
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
if [[ "${qec_group}" == d3_* ]]; then qec_shots=20000000; else qec_shots=60000000; fi
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.train \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --cache-dir "${QEC_ASSET_ROOT}/synthetic/${qec_group}/seed0-shots${qec_shots}" \
  --group "${qec_group}" --output "${QEC_ASSET_ROOT}/pilot/${qec_group}" \
  --stage pretrain --seed 0 --hours 0.4 --max-steps 1000 --eval-seconds 300

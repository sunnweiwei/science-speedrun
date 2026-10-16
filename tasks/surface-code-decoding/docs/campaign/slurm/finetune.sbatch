#!/usr/bin/env bash
#SBATCH --job-name=qec-finetune
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|v100
#SBATCH --exclude=gpu-compute5,linux[41-60]
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:15:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_CAMPAIGN:=native-v1}"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
qec_models="${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/models/${qec_group}"
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.train \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" --group "${qec_group}" \
  --output "${qec_models}/finetune" --init-model "${qec_models}/pretrain/best.keras" \
  --stage finetune --seed 0 --hours 2 --epochs 100 --eval-seconds 300

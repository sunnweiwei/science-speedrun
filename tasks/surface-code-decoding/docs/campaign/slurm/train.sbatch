#!/usr/bin/env bash
#SBATCH --job-name=qec-pretrain
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:30:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_TRAIN_HOURS:=24}"
: "${QEC_CAMPAIGN:=native-v1}"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
if [[ "${qec_group}" == d3_* ]]; then qec_shots=20000000; else qec_shots=60000000; fi
# --hours is cumulative, including earlier sessions of this stage. A restart
# resumes last.keras and the exact data cursor; best.keras remains validation-selected.
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.train \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --cache-dir "${QEC_ASSET_ROOT}/synthetic/${qec_group}/seed0-shots${qec_shots}" \
  --group "${qec_group}" \
  --output "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/models/${qec_group}/pretrain" \
  --stage pretrain --seed 0 --hours "${QEC_TRAIN_HOURS}" --epochs 100 \
  --eval-seconds 300

#!/usr/bin/env bash
#SBATCH --job-name=qec-score
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_CAMPAIGN:=native-v1}"
for qec_stage in pretrain finetune; do
  "${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.score_neural \
    --predictions "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/predictions/${qec_stage}" \
    --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
    --classical-root "${QEC_ASSET_ROOT}/classical-v1" \
    --output "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/scores-${qec_stage}.json" \
    --bootstrap-samples 2000 --seed 20260919
done
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.accounting --asset-root "${QEC_ASSET_ROOT}"
"${QEC_TASK_ROOT}/.venv/bin/python" -m baselines.report \
  --asset-root "${QEC_ASSET_ROOT}" --campaign "${QEC_CAMPAIGN}" \
  --output "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/report.html"

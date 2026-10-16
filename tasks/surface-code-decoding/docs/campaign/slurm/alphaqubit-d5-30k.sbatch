#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-d5-30k
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --array=8-9%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
qec_python="${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python"
qec_parent="${QEC_ASSET_ROOT}/alphaqubit-v1"
qec_followup="${QEC_ASSET_ROOT}/alphaqubit-d5-30k"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
case "${qec_group}" in d5_bX_center_5_5|d5_bZ_center_5_5) ;; *) exit 2 ;; esac
export PYTHONPATH="${QEC_TASK_ROOT}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${qec_parent}/jax-cache"
unset LD_LIBRARY_PATH
# Resume a separately copied state; the completed v1 campaign stays immutable.
# The 4.5-hour cumulative limit leaves about 30 minutes for freeze and inference.
"${qec_python}" -m baselines.alphaqubit_train \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" --group "${qec_group}" \
  --output "${qec_followup}/models/${qec_group}/finetune" \
  --stage finetune --seed 0 --hours 4.5 --max-steps 30000 \
  --dtype bfloat16 --microbatch 256 --batch-change-counter steps \
  --eval-shots 5120 --eval-seconds 600 --resume \
  --init-model "${qec_parent}/models/${qec_group}/pretrain/best.npz"
JAX_PLATFORMS=cpu "${qec_python}" -m baselines.alphaqubit_artifact \
  --models-root "${qec_followup}/models" --stage finetune --group "${qec_group}" \
  --output "${qec_followup}/packages/${qec_group}"
# Reuse the already installed locked inference environment, with package sources.
cd "${qec_followup}/packages/${qec_group}"
export PYTHONPATH="${qec_followup}/packages/${qec_group}"
"${qec_parent}/replay/both/.venv/bin/python" -m baselines.alphaqubit_predict \
  --package "${qec_followup}/packages/${qec_group}" \
  --data-root "${qec_parent}/replay-inputs" --split test --require-gpu \
  --group "${qec_group}" --stage finetune \
  --output "${qec_followup}/predictions/finetune/${qec_group}"

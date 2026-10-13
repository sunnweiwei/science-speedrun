#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-train
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:15:00
#SBATCH --signal=B:USR1@90
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="${QEC_ASSET_ROOT}/alphaqubit-v1/jax-cache"
unset LD_LIBRARY_PATH
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
qec_stage="${QEC_STAGE:-pretrain}"
qec_output="${QEC_ASSET_ROOT}/alphaqubit-v1/models/${qec_group}/${qec_stage}"
qec_extra=()
if [[ -f "${qec_output}/last.pkl" ]]; then qec_extra+=(--resume); fi
if [[ "${qec_stage}" == finetune ]]; then
  qec_extra+=(--init-model "${QEC_ASSET_ROOT}/alphaqubit-v1/models/${qec_group}/pretrain/best.npz")
fi
# B: targets the batch process; exec makes that process the Python handler.
# The cumulative training limit normally stops before this Slurm warning.
exec "${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python" -m baselines.alphaqubit_train \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" --group "${qec_group}" \
  --output "${qec_output}" --stage "${qec_stage}" --seed 0 \
  --hours "${QEC_TRAIN_HOURS:-2}" --dtype bfloat16 --microbatch 256 \
  --batch-change-counter steps --eval-shots 5120 --eval-seconds 600 \
  "${qec_extra[@]}"

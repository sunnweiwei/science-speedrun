#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-replay
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
qec_campaign="${QEC_ASSET_ROOT}/alphaqubit-v1"
qec_package="${qec_campaign}/replay/both"
cd "${qec_package}"
export PYTHONPATH="${qec_package}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${qec_campaign}/replay-jax-cache"
unset LD_LIBRARY_PATH
for qec_stage in pretrain finetune; do
  "${qec_package}/.venv/bin/python" -m baselines.alphaqubit_predict \
    --package "${qec_package}" --data-root "${qec_campaign}/replay-inputs" \
    --split test --group "${qec_group}" --stage "${qec_stage}" --require-gpu \
    --output "${qec_campaign}/predictions/${qec_stage}/${qec_group}"
done

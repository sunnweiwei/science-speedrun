#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-pilot
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="${QEC_ASSET_ROOT}/alphaqubit-v1/jax-cache"
unset LD_LIBRARY_PATH
qec_group=d3_bX_center_3_5
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 1 ]]; then qec_group=d5_bX_center_5_5; fi
mkdir -p "${QEC_ASSET_ROOT}/alphaqubit-v1/pilot"
exec "${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python" -m baselines.alphaqubit_pilot \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" --group "${qec_group}" \
  --dtype "${QEC_PILOT_DTYPE:-float32}" --batch-size "${QEC_PILOT_BATCH:-32}" \
  --output "${QEC_ASSET_ROOT}/alphaqubit-v1/pilot/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${qec_group}.json"

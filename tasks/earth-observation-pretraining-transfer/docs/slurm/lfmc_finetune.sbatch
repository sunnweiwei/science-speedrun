#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]]
: "${OLMO_PACKAGE:?}" "${OLMO_OUTPUT:?}"
export PYTHONPATH=${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/downstream-deps
export PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec ${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream/.venv/bin/python ${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer/docs/downstream/train_lfmc.py --index ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/lfmc-downstream-index.json --package "$OLMO_PACKAGE" --output "$OLMO_OUTPUT" --microbatch 2 --pilot

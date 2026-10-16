#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=lfmc-index
#SBATCH --output=earth-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
exec ${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream/.venv/bin/python ${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer/docs/downstream/lfmc_batch.py --root ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/lfmc-downstream-trainval --output ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/lfmc-downstream-index.json

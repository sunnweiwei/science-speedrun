#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=lfmc-external-metadata
#SBATCH --output=earth-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/downstream-deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec ${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream/.venv/bin/python ${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer/docs/downstream/external_lfmc_metadata.py --workbook ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/external-lfmc/globe-lfmc-2.0.xlsx --policy ${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer/docs/analysis/external-lfmc/metadata-selection-policy.json --pretrain-coordinates ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/pretrain-two-shards-filtered/latlon_distribution.npy ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/pretrain-filtered/latlon_distribution.npy --output ${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth/external-lfmc/metadata-candidates.json

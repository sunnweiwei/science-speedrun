#!/bin/bash
#SBATCH --job-name=olmoearth-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
: "${OLMO_MODEL:?exported package required}" "${OLMO_INPUT:?unlabeled H5 required}" "${OLMO_OUTPUT:?output directory required}"
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_PY=$OLMO_UPSTREAM/.venv/bin/python
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$OLMO_PY" "$OLMO_TRACK/tests/replay.py" --model "$OLMO_MODEL" --input "$OLMO_INPUT" --output "$OLMO_OUTPUT"

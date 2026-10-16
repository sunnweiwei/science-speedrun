#!/bin/bash
#SBATCH --job-name=olmoearth-sharded-infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
: "${OLMO_MODEL:?exported model required}" "${OLMO_SHARD_INPUTS:?input shards required}" "${OLMO_SHARD_OUTPUT:?output directory required}"
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]] || exit 2
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
printf -v OLMO_PART 'part%03d' "$SLURM_ARRAY_TASK_ID"
"$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/tests/replay.py" \
  --model "$OLMO_MODEL" --input "$OLMO_SHARD_INPUTS/$OLMO_PART.h5" \
  --output "$OLMO_SHARD_OUTPUT/$OLMO_PART"

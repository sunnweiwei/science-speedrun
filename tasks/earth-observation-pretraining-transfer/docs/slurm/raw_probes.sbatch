#!/bin/bash
#SBATCH --job-name=olmoearth-raw-probes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=earth-%j.log
# CPU-only scoring; request no GPUs. Pin the same CPU family as model probes via sbatch --nodelist if needed.
set -euo pipefail
OLMO_ROOT=${OLMO_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth}
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
for task in awf lfmc; do
  "$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/environment/evaluation/probes.py" \
    --features "$OLMO_ROOT/features/raw-$task.npz" --labels "$OLMO_ROOT/eval/$task/labels.json" \
    --task "$task" --validation-predictions "$OLMO_ROOT/results/large-raw-$task-predictions.npz" \
    --output "$OLMO_ROOT/results/large-raw-$task.json"
done

#!/bin/bash
#SBATCH --job-name=olmoearth-merge-probes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
: "${OLMO_ROOT:?evaluation root required}" "${OLMO_NAME:?model name required}" "${OLMO_SHARD_INPUTS:?input shards required}" "${OLMO_SHARD_OUTPUT:?shard features required}"
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
OLMO_FEATURES="$OLMO_ROOT/features/$OLMO_NAME-lfmc/features.npz"
"$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/docs/analysis/merge_features.py" \
  --manifest "$OLMO_SHARD_INPUTS/manifest.json" --features "$OLMO_SHARD_OUTPUT" --output "$OLMO_FEATURES"
"$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/environment/evaluation/probes.py" \
  --features "$OLMO_FEATURES" --labels "$OLMO_ROOT/eval/lfmc/labels.json" --task lfmc \
  --validation-predictions "$OLMO_ROOT/results/$OLMO_NAME-lfmc-predictions.npz" \
  --output "$OLMO_ROOT/results/$OLMO_NAME-lfmc.json"

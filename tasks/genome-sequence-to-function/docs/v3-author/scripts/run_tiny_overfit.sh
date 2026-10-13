#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MATERIALIZED_DATA_DIR OUTPUT_DIR [EPOCHS=120]" >&2
  exit 2
fi

data_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
epochs=${3:-120}
starter_dir=$(realpath "$(dirname "$0")/../..")
image=${ALPHAGENOME_IMAGE:-alphagenome-50m-dev:latest}

if [[ ! -s "$data_dir/release.json" || ! -s "$data_dir/public/train.parquet" ||
      ! -s "$data_dir/labels/train.parquet" ]]; then
  echo "materialized training data is incomplete: $data_dir" >&2
  exit 2
fi
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$output_dir"
docker_args=(
  run --rm --gpus all --network none --shm-size 4g
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=4
  -v "$data_dir:/data:ro"
  -v "$starter_dir:/workspace/starter:ro"
  -v "$output_dir:/output"
  -w /workspace/starter "$image"
)

docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval train \
  --input /data --output /output \
  --config /workspace/starter/reference/alphagenome_50m/tiny_overfit.yaml \
  --epochs "$epochs" --max-examples 1 --workers 0 --device cuda \
  --learning-rate 0.003 --min-learning-rate 0.003 --weight-decay 0 \
  --save-every 1000 --log-every 10 | tee "$output_dir/train.log"

docker "${docker_args[@]}" python -m reference.alphagenome_50m.tiny_overfit_report \
  --input /data --checkpoint /output/last.pt --report /output/tiny_overfit_report.json \
  --device cuda

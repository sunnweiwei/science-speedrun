#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 DATA_DIR OUTPUT_DIR [EPOCHS=10] [GPU_ID=0]" >&2
  exit 2
fi

DATA_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath -m "$2")
EPOCHS=${3:-10}
GPU_ID=${4:-0}
IMAGE=${ALPHAGENOME_IMAGE:-alphagenome-50m-dev}
STARTER_DIR=$(realpath "$(dirname "$0")/../..")

if [[ ! -s "$DATA_DIR/release.json" ]] || \
   [[ ! -s "$DATA_DIR/public/train.parquet" ]] || \
   [[ ! -s "$DATA_DIR/public/valid.parquet" ]]; then
  echo "complete validated AlphaGenome data required at $DATA_DIR" >&2
  exit 2
fi
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "EPOCHS must be positive and GPU_ID must be non-negative" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/pipeline.log") 2>&1
echo "RUNNING" > "$OUTPUT_DIR/pipeline.status"
pipeline_status=FAILED
trap 'echo "$pipeline_status" > "$OUTPUT_DIR/pipeline.status"' EXIT

docker_args=(
  run --rm --gpus "device=$GPU_ID"
  --network none
  --user "$(id -u):$(id -g)"
  --shm-size 16g
  -e HOME=/tmp
  -v "$STARTER_DIR:/workspace/starter:ro"
  -v "$DATA_DIR/reference:/data/reference:ro"
  -v "$DATA_DIR/public:/data/public:ro"
  -v "$DATA_DIR/labels/train.parquet:/data/labels/train.parquet:ro"
  -v "$DATA_DIR/labels/valid.parquet:/data/labels/valid.parquet:ro"
  -v "$DATA_DIR/scoring/train:/data/scoring/train:ro"
  -v "$DATA_DIR/scoring/valid:/data/scoring/valid:ro"
  -v "$DATA_DIR/targets/train:/data/targets/train:ro"
  -v "$DATA_DIR/targets/valid:/data/targets/valid:ro"
  -v "$OUTPUT_DIR:/output"
  -w /workspace/starter
  "$IMAGE"
)

best_score=-inf
start_epoch=1
if [[ -s "$OUTPUT_DIR/best_valid_metrics.json" ]]; then
  best_score=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall_mean_family_pearson"])' "$OUTPUT_DIR/best_valid_metrics.json")
fi
for report in "$OUTPUT_DIR"/valid_epoch_*.json; do
  [[ -e "$report" ]] || continue
  completed=${report##*valid_epoch_}; completed=${completed%.json}
  if [[ "$completed" =~ ^[0-9]+$ ]] && (( completed >= start_epoch )); then
    start_epoch=$((completed + 1))
  fi
done
for epoch in $(seq "$start_epoch" "$EPOCHS"); do
  resume_args=()
  if [[ -s "$OUTPUT_DIR/last.pt" ]]; then
    resume_args=(--resume /output/last.pt)
  fi
  docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval train \
    --input /data --output /output --epochs "$EPOCHS" --stop-after-epoch "$epoch" --workers 2 \
    --save-every 250 --log-every 25 --device cuda "${resume_args[@]}"
  docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval evaluate \
    --input /data --checkpoint /output/last.pt --split valid --workers 2 \
    --log-every 25 --report "/output/valid_epoch_${epoch}.json" --device cuda
  score=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall_mean_family_pearson"])' "$OUTPUT_DIR/valid_epoch_${epoch}.json")
  if python3 -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)' "$score" "$best_score"; then
    cp "$OUTPUT_DIR/last.pt" "$OUTPUT_DIR/best.pt"
    cp "$OUTPUT_DIR/valid_epoch_${epoch}.json" "$OUTPUT_DIR/best_valid_metrics.json"
    best_score=$score
  fi
done

cp "$OUTPUT_DIR/best_valid_metrics.json" "$OUTPUT_DIR/valid_metrics.json"

touch "$OUTPUT_DIR/PIPELINE_COMPLETE"
pipeline_status=COMPLETE

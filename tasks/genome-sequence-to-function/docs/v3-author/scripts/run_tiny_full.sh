#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MATERIALIZED_DATA_DIR OUTPUT_DIR [EPOCHS=3]" >&2
  exit 2
fi

data_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
epochs=${3:-3}
starter_dir=$(realpath "$(dirname "$0")/../..")
image=${ALPHAGENOME_IMAGE:-alphagenome-50m-dev:latest}
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi
for path in release.json public/train.parquet public/valid.parquet public/test.parquet \
            labels/train.parquet labels/valid.parquet labels/test.parquet; do
  if [[ ! -s "$data_dir/$path" ]]; then
    echo "materialized data is incomplete: $data_dir/$path" >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
docker_args=(
  run --rm --gpus all --network none --shm-size 4g
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=4
  -v "$data_dir:/data:ro"
  -v "$starter_dir:/workspace/starter:ro"
  -v "$output_dir:/output"
  -w /workspace/starter "$image"
)

for epoch in $(seq 1 "$epochs"); do
  if [[ ! -s "$output_dir/valid_epoch_${epoch}.json" ]]; then
    resume_args=()
    if [[ -s "$output_dir/last.pt" ]]; then
      resume_args=(--resume /output/last.pt)
    fi
    docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval train \
      --input /data --output /output \
      --config /workspace/starter/reference/alphagenome_50m/tiny_overfit.yaml \
      --epochs "$epochs" --stop-after-epoch "$epoch" --workers 0 --device cuda \
      --learning-rate 0.001 --min-learning-rate 0.0001 --weight-decay 0.0001 \
      --save-every 250 --log-every 100 "${resume_args[@]}" \
      | tee -a "$output_dir/train.log"
    cp "$output_dir/last.pt" "$output_dir/epoch_${epoch}.pt"
    docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval evaluate \
      --input /data --checkpoint /output/last.pt --split valid --workers 0 \
      --log-every 50 --report "/output/valid_epoch_${epoch}.json" --device cuda \
      | tee -a "$output_dir/validation.log"
    if [[ ! -s "$output_dir/best_valid_metrics.json" ]] || \
       python3 - "$output_dir/valid_epoch_${epoch}.json" "$output_dir/best_valid_metrics.json" <<'PY'
import json
import sys
scores = [json.load(open(path))["overall_mean_family_pearson"] for path in sys.argv[1:]]
raise SystemExit(0 if scores[0] > scores[1] else 1)
PY
    then
      cp "$output_dir/last.pt" "$output_dir/best.pt"
      cp "$output_dir/valid_epoch_${epoch}.json" "$output_dir/best_valid_metrics.json"
      echo "$epoch" > "$output_dir/best_epoch.txt"
    fi
  fi
done

if [[ ! -s "$output_dir/best.pt" || ! -s "$output_dir/best_valid_metrics.json" ]]; then
  echo "missing validation-selected checkpoint" >&2
  exit 1
fi

if [[ ! -s "$output_dir/heldout_test_metrics.json" ]]; then
  docker "${docker_args[@]}" python -m reference.alphagenome_50m.train_eval evaluate \
    --input /data --checkpoint /output/best.pt --split test --workers 0 \
    --log-every 100 --report /output/heldout_test_metrics.json --device cuda \
    | tee -a "$output_dir/test.log"
fi

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: fit.sh TRAIN_CSV VAL_CSV RUN_DIR [OMATG_OVERRIDES ...]" >&2
  exit 2
fi

TRAIN_CSV=$(realpath "$1")
VAL_CSV=$(realpath "$2")
RUN_DIR=$(realpath -m "$3")
shift 3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -e "$RUN_DIR/input" ]]; then
  echo "refusing to overwrite existing run input: $RUN_DIR/input" >&2
  exit 1
fi
mkdir -p "$RUN_DIR/input"
cp "$TRAIN_CSV" "$RUN_DIR/input/train.csv"
cp "$VAL_CSV" "$RUN_DIR/input/val.csv"

python3 "$SCRIPT_DIR/sampler_stats.py" \
  --train-csv "$RUN_DIR/input/train.csv" \
  --output "$RUN_DIR/sampler_stats.json"

# OMatG creates its CSV-to-LMDB cache beside the CSV.  The copies above keep
# the task's mounted source split read-only while giving preprocessing a safe
# run-local destination.
export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec omg fit \
  --config="$SCRIPT_DIR/config.yaml" \
  --data.train_dataset.init_args.file_path="$RUN_DIR/input/train.csv" \
  --data.val_dataset.init_args.file_path="$RUN_DIR/input/val.csv" \
  --data.pred_dataset.init_args.file_path="$RUN_DIR/input/val.csv" \
  --trainer.default_root_dir="$RUN_DIR" \
  --seed_everything=20260903 \
  "$@"

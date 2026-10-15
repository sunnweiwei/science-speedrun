#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: train.sh TRAIN_CSV SUBMISSION_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/train.py" \
  --train-csv "$1" \
  --submission-dir "$2" \
  --starter-root "$SCRIPT_DIR"

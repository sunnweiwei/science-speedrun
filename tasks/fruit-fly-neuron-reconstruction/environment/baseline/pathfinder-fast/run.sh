#!/usr/bin/env bash
set -euo pipefail
code="${PATHFINDER_CODE:-/opt/pathfinder/upstream/pathfinder}"
mode="${1:?Use train-volume, train-shape, export, publish, or predict}"; shift
case "$mode" in
  train-volume) script=train_volume.py ;;
  train-shape) script=train_shape.py ;;
  growth-reference) script=ffn_growth_validation.py ;;
  export) script=export_weights.py ;;
  publish) script=publish_weights.py ;;
  predict) script=single_pipeline.py ;;
  *) echo "Unknown baseline command: $mode" >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" -B "$code/$script" "$@"

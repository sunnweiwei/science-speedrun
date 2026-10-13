#!/usr/bin/env bash
# timeseries-forecast/v1 entry point: bash infer.sh INPUT_DIR OUTPUT_DIR
set -euo pipefail
exec python3 "$(dirname "$(readlink -f "$0")")/predict.py" "$1" "$2"

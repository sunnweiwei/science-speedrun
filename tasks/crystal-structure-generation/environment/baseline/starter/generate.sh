#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: generate.sh INPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/generate.py" \
  --model "$SCRIPT_DIR/model.json" \
  --input-root "$1" \
  --output-root "$2"

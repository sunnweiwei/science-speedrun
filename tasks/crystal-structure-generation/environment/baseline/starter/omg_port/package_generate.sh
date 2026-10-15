#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: generate.sh INPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$SCRIPT_DIR/omg_port/generation.py" \
  --package-root "$SCRIPT_DIR" \
  --input-root "$1" \
  --output-root "$2"

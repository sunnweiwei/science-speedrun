#!/usr/bin/env bash
set -euo pipefail
# Model files are found next to shared_infer.py; INPUT_DIR and OUTPUT_DIR stay relative to the caller.
exec python "$(dirname -- "$0")/shared_infer.py" "$1" "$2"

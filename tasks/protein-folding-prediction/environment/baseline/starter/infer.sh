#!/usr/bin/env bash
set -euo pipefail
model_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$model_directory/predict.py" "$@"

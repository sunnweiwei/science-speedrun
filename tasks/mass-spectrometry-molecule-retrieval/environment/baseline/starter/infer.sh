#!/usr/bin/env bash
set -euo pipefail
massspec_model_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$massspec_model_dir/predict.py" "$1" "$2"

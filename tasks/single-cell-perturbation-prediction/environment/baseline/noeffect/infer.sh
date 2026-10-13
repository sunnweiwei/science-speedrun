#!/usr/bin/env bash
# B0: no-effect baseline — resample basal cells unchanged. Same rule as the scorer's baseline.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python "$here/predict.py" "$1" "$2"

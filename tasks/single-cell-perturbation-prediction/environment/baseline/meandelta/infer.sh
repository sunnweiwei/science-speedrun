#!/usr/bin/env bash
# B1 (mode.json gated=false) / B1g (gated=true): cross-context mean delta on resampled basal cells.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec python "$here/predict.py" "$1" "$2"

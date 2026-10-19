#!/usr/bin/env bash
set -euo pipefail
# Keep the explicit local/HTTPS release interface for maintainer exports.
for arg in "$@"; do
  if [[ "$arg" == --manifest || "$arg" == --manifest=* ]]; then
    exec python3 "$(dirname "$0")/materialize.py" "$@"
  fi
done
exec python3 "$(dirname "$0")/download_hf.py" "$@"

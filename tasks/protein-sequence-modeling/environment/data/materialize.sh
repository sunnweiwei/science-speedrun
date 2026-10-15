#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: materialize.sh TARGET_DIRECTORY" >&2
  exit 64
fi
exec python3 "$(dirname "$0")/materialize.py" "$1"

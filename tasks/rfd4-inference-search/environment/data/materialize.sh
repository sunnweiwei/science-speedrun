#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/materialize.py" "${1:?Usage: materialize.sh DATA_ROOT}"

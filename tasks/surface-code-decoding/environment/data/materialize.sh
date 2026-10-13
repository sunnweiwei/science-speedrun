#!/usr/bin/env bash
set -euo pipefail
exec "${QEC_PYTHON:-python3}" "$(dirname "$0")/materialize.py" "$@"

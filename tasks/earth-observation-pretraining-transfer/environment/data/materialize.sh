#!/usr/bin/env bash
# Controller entry point; requires the pinned runtime from environment/setup.sh.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=''
exec "${OLMO_PY:-python}" "$HERE/materialize.py" "$@"

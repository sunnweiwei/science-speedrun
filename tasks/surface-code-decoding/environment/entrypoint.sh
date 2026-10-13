#!/usr/bin/env bash
set -euo pipefail
python /opt/qec/check_resources.py --data-root /workspace/data
exec "$@"

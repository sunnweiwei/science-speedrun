#!/usr/bin/env bash
# Development container entry: confirm the read-only data mount and the workspace layout, then hand over.
set -euo pipefail
if [ ! -f /workspace/data/materialize-receipt.json ]; then
  echo "WARNING: /workspace/data has no materialize-receipt.json; mount the prepared TASK_DATA_DIR entries under /workspace/data" >&2
fi
if [ -d /workspace/data/gifteval ]; then
  echo "ERROR: /workspace/data/gifteval (the evaluation series, including the test region) is mounted; the agent environment must only see gifteval_pretrain/ and valid/" >&2
  exit 1
fi
cp -f /workspace/evaluation/ts_io.py /workspace/baseline/starter/ts_io.py 2>/dev/null || true
exec "$@"

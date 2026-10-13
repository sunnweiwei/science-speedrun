#!/usr/bin/env bash
# Sanity-check the mounted release before handing the container to the agent.
set -euo pipefail
if [[ ! -s /workspace/data/manifest.json ]]; then
  echo "virtual-cell-zeroshot: /workspace/data is not a materialised release (manifest.json missing)." >&2
  echo "Mount the entries of TASK_DATA_DIR (the release's dev/) under /workspace/data (read-only)." >&2
  exit 66
fi
if [[ -e /workspace/data/private || -d /workspace/private ]]; then
  echo "virtual-cell-zeroshot: private responses must never be mounted into the development container." >&2
  exit 67
fi
exec "$@"

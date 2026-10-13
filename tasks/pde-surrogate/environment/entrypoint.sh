#!/usr/bin/env bash
# Development container entry: confirm the data mounts are the agent-side view, then hand over.
#   - /workspace/data must hold a materialization (materialize-receipt.json from the upstream pools and
#     release-receipt.json from `materialize.py --from-release --subtree train`);
#   - the corpus files the split cuts must be the truncated train heads (check_agent_mount.py).
set -euo pipefail
if [ ! -f /workspace/data/materialize-receipt.json ] || [ ! -f /workspace/data/release-receipt.json ]; then
  echo "WARNING: /workspace/data lacks materialize-receipt.json or release-receipt.json; mount the prepared agent-side data (TASK_DATA_DIR)" >&2
fi
python3 /opt/pde-surrogate/check_agent_mount.py /workspace/data /workspace/data/split.json
exec "$@"

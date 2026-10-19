#!/usr/bin/env bash
set -euo pipefail
# Fresh inference and scoring; infrastructure failures never create reward.json.
# The NVIDIA container runtime bind-mounts entries under /proc/driver/nvidia; the kernel then refuses
# the fresh procfs of the prediction's user namespace ("Can't mount proc"). Detach them first, as the
# weather verifier does; the GPUs stay usable through /dev/nvidia*.
for mount in $(awk '$5 ~ /^\/proc\/driver\/nvidia/ {print $5}' /proc/self/mountinfo | sort -r); do
  umount "${mount}" 2>/dev/null || echo "warning: could not detach ${mount}; the sandbox may fail" >&2
done
exec python3 -B /opt/pathfinder/upstream/pathfinder/verify_single.py \
  --submission /workspace/submission \
  --image /benchmark/image --suite /benchmark/suite --runtime /benchmark/runtime \
  --limits /opt/pathfinder/limits.json --root /workspace/research/run \
  --logs /logs/verifier --seconds 41400

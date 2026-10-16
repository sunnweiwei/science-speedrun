#!/bin/bash
set -euo pipefail
OLMO_TEST_DIR=$(cd "$(dirname "$0")" && pwd)
# As in the weather verifier: the NVIDIA container runtime bind-mounts entries under
# /proc/driver/nvidia, and the kernel then refuses the fresh procfs of the model's user namespace
# ("Can't mount proc on /newroot/proc"). Detach them; the GPUs stay usable through /dev/nvidia*.
for mount in $(awk '$5 ~ /^\/proc\/driver\/nvidia/ {print $5}' /proc/self/mountinfo | sort -r); do
  umount "${mount}" 2>/dev/null || echo "warning: could not detach ${mount}; the sandbox may fail" >&2
done
# Docker exposes the allocated GPUs without setting CUDA_VISIBLE_DEVICES; name them explicitly.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]] && command -v nvidia-smi >/dev/null; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd,)
  export CUDA_VISIBLE_DEVICES
fi
OLMO_PROBES=${OLMO_PROBES:-$OLMO_TEST_DIR/../environment/evaluation/probes.py}
[[ -f "$OLMO_PROBES" ]] || OLMO_PROBES=/evaluation/probes.py
python "$OLMO_TEST_DIR/grade.py" --model "${OLMO_MODEL:-/workspace/submission}" --data "${OLMO_EVAL_DATA:-/data/eval}" --output "${OLMO_VERIFIER_OUTPUT:-/logs/verifier}" --probes "$OLMO_PROBES"

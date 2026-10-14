#!/usr/bin/env bash
# Independent WN2 model replay; the model never receives scoring truth mounts.
set -euo pipefail
WN2_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WN2_LOG=${WEATHER_VERIFIER_LOG_DIR:-/logs/verifier}
WN2_BENCHMARK=${WEATHER_WN2_BENCHMARK:-/benchmark}
WN2_EVAL_PYTHON=${WEATHER_EVAL_PYTHON:-/opt/wn2-eval/bin/python}
mkdir -p "${WN2_LOG}"
trap 'printf "{\"reward\": 0.0}\n" > "${WN2_LOG}/reward.json"; printf "{\"status\":\"failed\",\"failure_stage\":\"verifier_setup_or_execution\"}\n" > "${WN2_LOG}/status.json"' ERR
export PYTHONPATH=${WEATHER_WBX_SOURCE:-/opt/weatherbenchX}
export WEATHER_MODEL_PYTHON=${WEATHER_MODEL_PYTHON:-/opt/weather-python/bin/python}
export OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The NVIDIA container runtime bind-mounts one /proc/driver/nvidia/gpus/<pci-id> per visible GPU.
# Those mounts are locked in the user namespace that namespace_inference.py creates, so the kernel
# refuses its fresh procfs ("Can't mount proc on /newroot/proc: Operation not permitted") and every
# case scores 0. Detaching them here keeps the sandbox's own /proc intact - the alternative, binding
# the container's /proc into the namespace, would hand the submission a view of the verifier. The
# GPUs stay fully usable: the devices are /dev/nvidia*, not these procfs entries.
for mount in $(awk '$5 ~ /^\/proc\/driver\/nvidia/ {print $5}' /proc/self/mountinfo | sort -r); do
  umount "${mount}" 2>/dev/null || echo "warning: could not detach ${mount}; the sandbox may fail" >&2
done
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd,)
  export CUDA_VISIBLE_DEVICES
fi
"${WN2_EVAL_PYTHON}" "${WN2_CODE}/verify_track.py" \
  --package "${WEATHER_SUBMISSION_DIR:-/workspace/submission}" \
  --benchmark "${WN2_BENCHMARK}" \
  --output "${WN2_LOG}/replay" \
  --budget-seconds "${WEATHER_EVALUATION_BUDGET_SECONDS:-43200}" \
  --disk-allowance-gb "${WEATHER_EVALUATION_DISK_ALLOWANCE_GB:-2000}" \
  --disk-reserve-gb "${WEATHER_EVALUATION_DISK_RESERVE_GB:-25}" \
  --concurrent-cases "${WEATHER_EVALUATION_CONCURRENT_CASES:-0}" \
  --score-workers "${WEATHER_EVALUATION_SCORE_WORKERS:-0}"
cp "${WN2_LOG}/replay/reward.json" "${WN2_LOG}/reward.json"
cp "${WN2_LOG}/replay/status.json" "${WN2_LOG}/status.json"

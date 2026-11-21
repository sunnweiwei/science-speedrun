#!/usr/bin/env bash
set -euo pipefail
# NVIDIA's injected proc entries can prevent mounting proc in the worker's PID namespace.
while IFS= read -r mount; do umount "$mount"; done < <(
  awk '$5 ~ /^\/proc\/driver\/nvidia(\/|$)/ {print $5}' /proc/self/mountinfo | sort -ru
)
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/details.json
args=(--program-track --private-seed /private/seed --reference "${RFD4_REFERENCE:-stochastic-tsr}" --seconds-per-slot "${RFD4_SECONDS_PER_SLOT:-36}")
if [[ -f /private/suite.json ]]; then args+=(--suite /private/suite.json); fi
exec python /workspace/evaluation/run.py "${args[@]}" \
  --work "$(mktemp -d)" --output /logs/verifier/reward.json

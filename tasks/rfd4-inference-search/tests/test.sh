#!/usr/bin/env bash
set -euo pipefail
# NVIDIA's injected proc entries can prevent mounting proc in the worker's PID namespace.
while IFS= read -r mount; do umount "$mount"; done < <(
  awk '$5 ~ /^\/proc\/driver\/nvidia(\/|$)/ {print $5}' /proc/self/mountinfo | sort -ru
)
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/details.json
rm -rf /logs/verifier/figure2
stage=$(mktemp -d /logs/verifier/.paired-report.XXXXXX)
trap 'rm -rf "$stage"' EXIT
args=(--program-track --private-seed /private/seed --reference "${RFD4_REFERENCE:-stochastic-tsr}" --seconds-per-slot "${RFD4_SECONDS_PER_SLOT:-36}")
if [[ -f /private/suite.json ]]; then args+=(--suite /private/suite.json); fi
python /workspace/evaluation/run.py "${args[@]}" \
  --work "$(mktemp -d)" --output "$stage/reward.json"
digest=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("submission_sha256", ""))' "$stage/details.json")
if [[ -n "$digest" ]]; then
  figure_args=(--receipts /private/figure2 --submission-sha256 "$digest" --output "$stage/figure2")
  if [[ -f /private/figure2/baselines.csv ]]; then figure_args+=(--baseline-targets /private/figure2/baselines.csv); fi
  python /opt/rfd4-overall/measured_report.py "${figure_args[@]}"
  mv "$stage/figure2" /logs/verifier/
fi
mv "$stage/details.json" /logs/verifier/
mv "$stage/reward.json" /logs/verifier/

#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/details.json
exec python /workspace/evaluation/run.py --private-seed /private/seed \
  --work "$(mktemp -d)" --output /logs/verifier/reward.json

#!/usr/bin/env bash
set -euo pipefail
exec python /tests/main_verifier.py \
  --submission /workspace/submission \
  --input /benchmark/test \
  --sealed /sealed/run \
  --output /logs/verifier \
  --timeout "${ATOMISTIC_MATERIALS_INFERENCE_TIMEOUT_SECONDS:-10800}"

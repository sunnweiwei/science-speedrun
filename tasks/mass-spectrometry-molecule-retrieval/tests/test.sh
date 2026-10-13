#!/usr/bin/env bash
set -euo pipefail
exec python /tests/main_verifier.py \
  --submission /workspace/submission \
  --input /benchmark \
  --sealed /sealed/run \
  --output /logs/verifier \
  --timeout "${MASSSPECGYM_INFERENCE_TIMEOUT_SECONDS:-18000}"

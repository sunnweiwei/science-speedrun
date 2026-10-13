#!/usr/bin/env bash
# Harbor verifier entry point (main container). The grader container runs /tests/grader.py.
set -euo pipefail
exec python /tests/main_verifier.py \
  --submission /workspace/submission \
  --benchmark /benchmark \
  --sealed /sealed/run \
  --output /logs/verifier \
  --timeout-per-context "${VCZ_INFERENCE_TIMEOUT_SECONDS:-3600}"

#!/usr/bin/env bash
set -euo pipefail
exec python3 /tests/main_verifier.py \
  --submission /workspace/submission \
  --input /benchmark \
  --sealed /sealed/run \
  --output /logs/verifier \
  --timeout "${PROTEIN_DESIGN_INFERENCE_TIMEOUT_SECONDS:-21600}"

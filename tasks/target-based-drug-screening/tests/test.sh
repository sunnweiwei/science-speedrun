#!/usr/bin/env bash
set -euo pipefail
exec python /tests/grade.py \
  --submission /workspace/submission \
  --data-root /private/benchmark \
  --output /logs/verifier

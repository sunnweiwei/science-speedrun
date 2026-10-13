#!/usr/bin/env bash
# Harbor verification entry point. The evaluation series are bound read-only at /private/data/gifteval (a
# root-only directory, so a package running as the runner uid cannot regenerate the labels), the submission
# at /workspace/submission; the split manifest is baked into /tests.
set -euo pipefail
exec python3 /tests/main_verifier.py \
  --submission /workspace/submission \
  --data "${TIMESERIES_VERIFIER_DATA:-/private/data}" \
  --manifest "${TIMESERIES_SPLIT_MANIFEST:-/tests/split.json}" \
  --output /logs/verifier \
  --timeout "${TIMESERIES_INFERENCE_TIMEOUT_SECONDS:-14400}"

#!/usr/bin/env bash
# Harbor verification entry point. Data are bound read-only at /private/data (a root-only directory,
# so a package running as the runner uid cannot regenerate the truth from it), the submission at
# /workspace/submission; the split manifests are baked into /tests. PDE_SURROGATE_SPLIT_MANIFEST
# selects /tests/split_smoke.json for a plumbing run on the reduced release.
set -euo pipefail
exec python3 /tests/main_verifier.py \
  --submission /workspace/submission \
  --data "${PDE_SURROGATE_VERIFIER_DATA:-/private/data}" \
  --manifest "${PDE_SURROGATE_SPLIT_MANIFEST:-/tests/split.json}" \
  --output /logs/verifier \
  --timeout "${PDE_SURROGATE_INFERENCE_TIMEOUT_SECONDS:-14400}"

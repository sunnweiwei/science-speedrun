#!/usr/bin/env bash
# Verifier entry point (main container): fine-tune the package on the 24 tasks, seal the predictions, and copy the
# grader's reward.json / report.json to /logs/verifier.
set -euo pipefail
test ! -e /logs/verifier/reward.json
exec python3 /tests/main_verifier.py --submission /workspace/submission --benchmark /benchmark \
  --sealed /sealed/run --output /logs/verifier

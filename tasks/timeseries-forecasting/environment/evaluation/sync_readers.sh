#!/usr/bin/env bash
# Copy ts_io.py into every package that must be self-contained (the verifier refuses symlinks that leave a package).
# Run after editing environment/evaluation/ts_io.py; tests/test_task.py fails if a copy has drifted.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
for p in environment/baseline/seasonal_naive docs/validation/released_models/trivial_context_quantiles environment/baseline/starter; do
  cp environment/evaluation/ts_io.py "$p/ts_io.py"
  echo "synced $p/ts_io.py"
done

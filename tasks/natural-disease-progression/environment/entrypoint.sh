#!/usr/bin/env bash
set -euo pipefail

test -f /workspace/references/scaffold/data.py || { echo "scaffold reference is missing" >&2; exit 66; }
test -f /workspace/baseline/temporal-gru/longitudinal.py || { echo "temporal-gru baseline is missing" >&2; exit 66; }
test -f /workspace/references/delphi/train.py || { echo "Delphi reference is missing" >&2; exit 66; }
exec "$@"

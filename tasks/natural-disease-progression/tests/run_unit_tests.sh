#!/usr/bin/env bash
set -euo pipefail
if [[ -d /task/tests ]]; then
  exec python3 -m pytest -q -p no:cacheprovider /task/tests
fi
cd "$(dirname "$0")/.."
exec python3 -m pytest -q -p no:cacheprovider tests

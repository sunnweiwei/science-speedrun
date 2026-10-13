#!/usr/bin/env bash
# Unit tests without data or GPU: python3 -m pytest tests/  (from the task directory)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
exec python3 -m pytest -q tests/ "$@"

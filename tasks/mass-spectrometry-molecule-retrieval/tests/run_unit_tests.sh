#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The tests need Python >= 3.11 (tomllib); the images' default python3 is 3.10, the GLACIER runtime is 3.11.
PY=${PYTHON:-python3}
"$PY" -c 'import tomllib' 2>/dev/null || PY=/opt/glacier/bin/python
exec "$PY" -m unittest discover -v -s tests -p 'test_*.py'

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: materialize.sh OUTPUT_DIR [materialize.py options]" >&2
  exit 2
fi

OUTPUT_DIR=$1
shift
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCORER_PYTHON=${CRYSTALGEN_SCORER_PYTHON:-/opt/lemat/.venv/bin/python}
if [[ ! -x "$SCORER_PYTHON" ]]; then
  SCORER_PYTHON=python3
fi
"$SCORER_PYTHON" "$SCRIPT_DIR/materialize.py" --output-root "$OUTPUT_DIR" "$@"
"$SCORER_PYTHON" "$SCRIPT_DIR/validate.py" "$OUTPUT_DIR"

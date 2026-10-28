#!/usr/bin/env bash
# Usage: bash environment/data/materialize.sh release ROOT
#          install the pinned prepared releases into ROOT/runtime/{development,private} (default)
#        bash environment/data/materialize.sh TRACK --source DIR --destination DIR
#          rebuild one track from its source (see MATERIALIZE.md)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == release ]]; then
  [[ $# -eq 2 ]] || { sed -n '2,5p' "$0" >&2; exit 2; }
  exec python "$HERE/release.py" install --root "$2"
fi
exec python "$HERE/materialize.py" "$@"

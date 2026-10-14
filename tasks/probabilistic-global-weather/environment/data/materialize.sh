#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ "${1:-}" == "--upstream" ]]; then
    shift
    exec python3 "$script_dir/materialize.py" "$@"
fi
exec python3 "$script_dir/download_hf.py" "$@"

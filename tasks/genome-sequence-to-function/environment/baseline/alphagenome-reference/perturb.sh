#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: perturb.sh INPUT_ROOT OUTPUT_ROOT" >&2; exit 2; }
package_dir=$(cd "$(dirname "$0")" && pwd)
[[ -s "$package_dir/last.pt" ]] || { echo "missing trained last.pt" >&2; exit 2; }
exec python "$package_dir/perturb.py" "$1" "$2" --checkpoint "$package_dir/last.pt"

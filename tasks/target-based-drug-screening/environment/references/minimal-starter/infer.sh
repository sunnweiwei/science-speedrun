#!/usr/bin/env bash
set -euo pipefail
starter_dir=$(cd -- "$(dirname -- "$0")" && pwd)
[[ $# -ge 2 ]] || {
  echo "usage: infer.sh INPUT_ROOT OUTPUT_ROOT [predict.py options]" >&2
  exit 2
}
exec python "$starter_dir/predict.py" "$1" "$2" --model-package "$starter_dir" "${@:3}"

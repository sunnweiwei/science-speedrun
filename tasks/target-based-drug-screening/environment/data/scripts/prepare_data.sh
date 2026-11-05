#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_code=$(cd -- "$script_dir/.." && pwd)

exec bash "$data_code/materialize.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Bootstrap the image's Python libraries without loading CUDA solver modules.
# The Python dispatcher selects the solver runtime separately for each case.
exec bash "$root/native_runtime.sh" jax python "$root/evaluate_native.py" "$@"

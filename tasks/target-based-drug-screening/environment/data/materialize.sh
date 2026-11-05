#!/usr/bin/env bash
set -euo pipefail

default_target=/home/zifengw2/github/rsi-for-science/.agent-runtime/target-based-drug-screening/materialized
target=${TARGET_BASED_DRUG_SCREENING_DATA_DIR:-${DRUGCLIP_DATA_DIR:-$default_target}}

# An explicit positional target remains supported for portable or scratch runs.
# If the first argument is an option, use the persistent runtime target.
if [[ $# -gt 0 && $1 != --* ]]; then
  target=$1
  shift
fi
if [[ $target != /* ]]; then
  echo "materialization target must be absolute for cross-worktree reuse: $target" >&2
  exit 64
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m drugclip_data.cli release --data-root "$target" "$@"

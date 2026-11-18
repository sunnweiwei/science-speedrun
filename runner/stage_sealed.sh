#!/usr/bin/env bash
# Put a task's sealed verifier files in place before verification and check them.
#   stage_sealed.sh TASK_DIR
# A task lists its sealed files in tests/sealed/SHA256SUMS. A file missing from the checkout is taken
# from SEALED_DIR when set, either another checkout (SEALED_DIR/TASK/tests/sealed/FILE) or a flat copy
# (SEALED_DIR/TASK/FILE). Verification stops when a file is missing or differs, so a verifier never
# runs on substitute seeds or rosters.
set -euo pipefail
dir=$1; task=$(basename "$dir"); sealed="$dir/tests/sealed"
[[ -f "$sealed/SHA256SUMS" ]] || exit 0
while read -r _ file; do
  [[ -n "$file" && ! -f "$sealed/$file" && -n "${SEALED_DIR:-}" ]] || continue
  for src in "$SEALED_DIR/$task/tests/sealed/$file" "$SEALED_DIR/$task/$file"; do
    if [[ -f "$src" ]]; then cp "$src" "$sealed/$file"; break; fi
  done
done < "$sealed/SHA256SUMS"
if ! (cd "$sealed" && sha256sum --quiet -c SHA256SUMS 2>/dev/null); then
  echo "$task: sealed verifier files are missing or differ from tests/sealed/SHA256SUMS." >&2
  echo "Set SEALED_DIR to a checkout of the private task repository; see $sealed/README.md." >&2
  exit 2
fi

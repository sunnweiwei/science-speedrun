#!/usr/bin/env bash
# Unit tests in the task's runtime image (CPU only).
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${ATOMISTIC_TEST_IMAGE:-atomistic-materials-env:v2}
[[ "${ATOMISTIC_TEST_SKIP_BUILD:-0}" == 1 ]] || docker build -t "$image" "$task_root/environment"
exec docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$task_root:/task:ro" -w /task "$image" \
  python -m pytest -q -p no:cacheprovider tests "$@"

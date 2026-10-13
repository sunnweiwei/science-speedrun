#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
test_python=${GENOME_SEQUENCE_TO_FUNCTION_TEST_PYTHON:-${ALPHAGENOME_TEST_PYTHON:-$script_dir/../.venv/bin/python}}
if [[ -x "$test_python" ]]; then
  exec "$test_python" -m pytest -q -p no:cacheprovider "$script_dir"
fi
command -v docker >/dev/null || {
  echo "ERROR: set GENOME_SEQUENCE_TO_FUNCTION_TEST_PYTHON or install Docker" >&2
  exit 2
}
task_root=$(cd "$script_dir/.." && pwd)
image=${ALPHAGENOME_MATERIALIZER_IMAGE:-genome-sequence-to-function-materializer:test}
docker build -f "$task_root/environment/data/materializer.Dockerfile" -t "$image" "$task_root"
exec docker run --rm --entrypoint python -e PYTHONPATH=/workspace/environment/data/src \
  -v "$task_root:/workspace:ro" -w /workspace "$image" -m pytest -q -p no:cacheprovider tests

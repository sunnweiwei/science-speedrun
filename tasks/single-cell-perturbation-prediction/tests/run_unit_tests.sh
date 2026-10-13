#!/usr/bin/env bash
# Run the unit tests inside the verifier image (needs cell-eval): bash tests/run_unit_tests.sh [image]
set -euo pipefail
task=$(cd "$(dirname "$0")/.." && pwd)
image=${1:-vcz-verifier:latest}
${VCZ_DOCKER:-docker} run --rm -v "$task:/task:ro" -w /task --entrypoint python "$image" -m pytest -q /task/tests/test_task.py

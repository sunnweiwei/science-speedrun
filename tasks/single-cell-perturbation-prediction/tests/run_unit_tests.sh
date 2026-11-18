#!/usr/bin/env bash
# Run the unit tests inside the verifier image (needs cell-eval): bash tests/run_unit_tests.sh [image]
set -euo pipefail
task=$(cd "$(dirname "$0")/.." && pwd)
image=${1:-vcz-verifier:latest}
${VCZ_DOCKER:-docker} run --rm -v "$task:/task:ro" -w /task --entrypoint python "$image" -m pytest -q /task/tests/test_task.py /task/tests/test_prediction_safety.py \
  /task/tests/test_query_independence.py /task/tests/test_calibrated_reward_v2.py /task/tests/test_wmse_v2_grader.py

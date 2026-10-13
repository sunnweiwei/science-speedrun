#!/usr/bin/env bash
set -euo pipefail
exec /opt/qec/.venv/bin/python /opt/task/tests/main_verifier.py --timeout "${QEC_INFERENCE_TIMEOUT_SECONDS:-3600}"

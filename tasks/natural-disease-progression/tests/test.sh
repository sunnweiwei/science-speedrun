#!/usr/bin/env bash
set -euo pipefail
exec python3 /tests/main_verifier.py --timeout "${NDP_INFERENCE_TIMEOUT_SECONDS:-42000}"

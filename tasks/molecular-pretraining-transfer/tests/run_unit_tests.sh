#!/usr/bin/env bash
# Unit tests (standard library; Python >= 3.11): layout, metrics, package checks, grader.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest -v test_task

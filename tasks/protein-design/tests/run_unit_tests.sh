#!/usr/bin/env bash
# Interface and metric tests with synthetic fixtures. No weights, no downloads.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m unittest discover -s . -p 'test_*.py' -v

#!/usr/bin/env bash
# Data-free unit and verifier-level tests. Requires python3 with h5py, numpy, pytest.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
python3 -m pytest tests -q "$@"

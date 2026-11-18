#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""
exec python3 -m unittest -v test_task test_v4_protein_sequence

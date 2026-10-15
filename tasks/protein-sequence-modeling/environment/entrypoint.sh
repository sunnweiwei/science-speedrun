#!/usr/bin/env bash
set -euo pipefail
test -s /workspace/data/manifest.json || { echo "materialized task data is missing" >&2; exit 66; }
test -s /workspace/data/pretrain/uniref50_subset.fasta || { echo "pretraining corpus is missing; run build_pretrain_subset.py" >&2; exit 66; }
exec "$@"

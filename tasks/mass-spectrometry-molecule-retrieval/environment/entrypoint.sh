#!/usr/bin/env bash
set -euo pipefail

test -s /workspace/data/manifest.json || {
  echo "materialized MassSpecGym development manifest is missing" >&2
  exit 66
}
test -s /workspace/data/train.jsonl || {
  echo "materialized MassSpecGym training fold is missing" >&2
  exit 66
}
test -s /workspace/data/validation.jsonl || {
  echo "materialized MassSpecGym validation fold is missing" >&2
  exit 66
}
test -s /workspace/data/synthetic_augmentation/manifest.json || {
  echo "held-out-clean synthetic augmentation is missing" >&2
  exit 66
}

python /workspace/data/check_resources.py --data-root /workspace/data
exec "$@"

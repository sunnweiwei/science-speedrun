#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$here/train.py" --stage both \
  --input-root /workspace/data \
  --work-dir /workspace/research/drugclip-short \
  --output-dir /workspace/research/candidate \
  --profsa-epochs 1 --finetune-epochs 1 \
  --batch-size 2 --validation-batch-size 4 --workers 0 \
  --max-pretrain-records 8 --max-train-records 8 \
  --max-validation-records 8 "$@"

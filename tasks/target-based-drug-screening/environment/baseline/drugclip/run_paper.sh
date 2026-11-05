#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 DATA_ROOT RUN_ROOT MODEL_OUTPUT" >&2
  exit 2
fi
DATA_ROOT=$(realpath "$1")
RUN_ROOT=$(realpath -m "$2")
MODEL_OUTPUT=$(realpath -m "$3")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_ROOT/profsa" "$RUN_ROOT/finetune"

if [[ -e "$RUN_ROOT/profsa/last.pt" || -e "$RUN_ROOT/finetune/last.pt" ]]; then
  echo "Existing optimizer checkpoints require explicit --resume; see README." >&2
  exit 2
fi

# Contrastive candidates are local to each GPU's batch of 48.
torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train.py" \
  --stage profsa --input-root "$DATA_ROOT" \
  --work-dir "$RUN_ROOT/profsa" --output-dir "$MODEL_OUTPUT" \
  --batch-size 48 --profsa-epochs 100

torchrun --standalone --nproc_per_node=4 "$SCRIPT_DIR/train.py" \
  --stage finetune --input-root "$DATA_ROOT" \
  --work-dir "$RUN_ROOT/finetune" --output-dir "$MODEL_OUTPUT" \
  --profsa-checkpoint "$RUN_ROOT/profsa/profsa_complete.pt" \
  --batch-size 48 --validation-batch-size 128 --finetune-epochs 200

#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
smoke_out="${DELPHI_SMOKE_OUTPUT:-/tmp/delphi-smoke}"
cd "$script_dir"
exec python train.py config/train_delphi_demo.py \
  --device=cpu \
  --out_dir="$smoke_out" \
  --batch_size=2 \
  --block_size=16 \
  --n_layer=1 \
  --n_head=2 \
  --n_embd=16 \
  --max_iters=0 \
  --eval_interval=1 \
  --log_interval=1 \
  --warmup_iters=1 \
  --lr_decay_iters=1


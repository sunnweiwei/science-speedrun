#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: score.sh INPUT_DIR GENERATED_DIR SCORE_DIR" >&2
  exit 2
fi

export HF_HOME=/scorer-assets/huggingface
export HF_HUB_CACHE=/scorer-assets/huggingface/hub
export HF_DATASETS_CACHE=/scorer-assets/huggingface/datasets
export XDG_CACHE_HOME=/scorer-assets/xdg-cache
export TORCH_HOME=/scorer-assets/torch-cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec /opt/lemat/.venv/bin/python /workspace/evaluation/score.py \
  --input "$1" \
  --generated "$2" \
  --output "$3"

#!/usr/bin/env bash
# Run one resumable family/seed joint-training job and full independent scoring.
set -euo pipefail
family=${1:?Usage: run_shared_full.sh FAMILY SEED}
seed=${2:?Usage: run_shared_full.sh FAMILY SEED}
case "$family" in eegnet|contrawr|cbramod|biot|codebrain) ;; *) exit 2;; esac
case "$seed" in 0|1|2|3|4) ;; *) exit 2;; esac
: "${TASK_DATA_DIR:?}"
: "${TASK_PRIVATE_DIR:?}"
: "${TASK_PRETRAINED_DIR:=$TASK_DATA_DIR/pretrained}"
: "${TASK_RESEARCH_DIR:?}"
python_bin=${BCI_PYTHON:-python}
root="$TASK_RESEARCH_DIR/shared-full"
run="$root/$family-s$seed"
mkdir -p "$run/score"
task=$(cd "$(dirname "$0")/../.." && pwd)
"$python_bin" "$task/tests/campaign/prepare_shared_campaign.py" --data "$TASK_DATA_DIR" \
  --private "$TASK_PRIVATE_DIR" --output "$root/campaign.json"
"$python_bin" - "$root/campaign.json" "${TASK_TRAIN_DATA_DIR:-$TASK_DATA_DIR}" "$task/environment/evaluation" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[3])
from data_identity import split_identity
campaign=json.loads(Path(sys.argv[1]).read_text())
data=Path(sys.argv[2])
for track,splits in campaign['development_identity'].items():
    for split,want in splits.items():
        got=split_identity(data/track/split)
        if got!=want:raise ValueError(f'{track}/{split}: staged data identity mismatch')
PY
extra=()
if [[ "$family" == cbramod || "$family" == biot || "$family" == codebrain ]]; then
  extra=(--checkpoint-dir "$TASK_PRETRAINED_DIR")
fi
if "$python_bin" "$task/environment/baseline/braindecode/train_shared.py" \
  --model "$family" --seed "$seed" --epochs 50 --steps-per-epoch 1000 \
  --batch-size 8 --lr 0.0001 --weight-decay 0.05 --num-workers 2 \
  --data "${TASK_TRAIN_DATA_DIR:-$TASK_DATA_DIR}" --output "$run/submission" "${extra[@]}" \
  >"$run/train.log" 2>&1; then
  echo 0 >"$run/train.exit"
else
  echo 1 >"$run/train.exit"
  exit 1
fi
if docker run --rm --network none --gpus all \
  -v "$TASK_PRIVATE_DIR:/private/data:ro" \
  -v "$run/submission:/workspace/submission:ro" \
  -v "$run/score:/logs/verifier" bci-verifier:0.2 \
  bash /tests/test.sh >"$run/score.log" 2>&1; then
  echo 0 >"$run/score.exit"
else
  echo 1 >"$run/score.exit"
  exit 1
fi
"$python_bin" - "$task/tests/campaign" "$run/score" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from check_receipt import checked_reward
for track in ('faced','tuev','ad65','openbmi','longitudinal'):
    checked_reward(sys.argv[2],track)
PY

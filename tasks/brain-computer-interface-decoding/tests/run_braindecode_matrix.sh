#!/usr/bin/env bash
# Five shared-model smoke runs. Each package is independently scored on all tracks.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${TASK_DATA_DIR:?Set TASK_DATA_DIR}"
: "${TASK_PRIVATE_DIR:?Set TASK_PRIVATE_DIR}"
: "${TASK_RESEARCH_DIR:?Set TASK_RESEARCH_DIR}"
: "${TASK_PRETRAINED_DIR:=$TASK_DATA_DIR/pretrained}"
root="$TASK_RESEARCH_DIR/shared-smoke"
mkdir -p "$root"
failures=0
for model in eegnet contrawr cbramod biot codebrain; do
  work="$root/$model"
  mkdir -p "$work"
  extra=()
  if [[ "$model" == cbramod || "$model" == biot || "$model" == codebrain ]]; then
    extra=(--checkpoint-dir /workspace/data/pretrained)
  fi
  if [[ -f "$work/train.exit" && "$(cat "$work/train.exit")" == 0 && -f "$work/submission/config.json" ]]; then
    python3 - "$work/submission/config.json" <<'PYCODE'
import json,sys
if json.load(open(sys.argv[1])).get('pipeline_version') != 'shared-v2.1-channel-preserving':
    raise ValueError('stale smoke package; archive it before rerunning')
PYCODE
    trained=0
  elif [[ -n "${BCI_PYTHON:-}" && -n "${TASK_TRAIN_DATA_DIR:-}" ]]; then
    host_extra=()
    if ((${#extra[@]})); then host_extra=(--checkpoint-dir "$TASK_PRETRAINED_DIR"); fi
    if "$BCI_PYTHON" "$TASK_ROOT/environment/baseline/braindecode/train_shared.py" \
      --model "$model" --data "$TASK_TRAIN_DATA_DIR" --output "$work/submission" \
      --seed 3407 --epochs 1 --steps-per-epoch 100 --batch-size 8 \
      --train-limit 1024 --val-limit 512 "${host_extra[@]}" >"$work/train.log" 2>&1; then
      trained=0
    else
      trained=1
    fi
  else
    if docker run --rm --network none --gpus all \
      -v "$TASK_DATA_DIR:/workspace/data:ro" \
      -v "$work:/workspace/research" "${BCI_DEVELOPMENT_IMAGE:-bci-development:0.2}" \
      python /workspace/baseline/braindecode/train_shared.py --model "$model" \
        --data /workspace/data --output /workspace/research/submission \
        --seed 3407 --epochs 1 --steps-per-epoch 100 --batch-size 8 \
        --train-limit 1024 --val-limit 512 "${extra[@]}" >"$work/train.log" 2>&1; then
      trained=0
    else
      trained=1
    fi
  fi
  echo "$trained" >"$work/train.exit"
  if [[ "$trained" == 0 ]] && \
  docker run --rm --network none --gpus all \
    -v "$TASK_PRIVATE_DIR:/private/data:ro" \
    -v "$work/submission:/workspace/submission:ro" \
    -v "$work:/logs/verifier" "${BCI_VERIFIER_IMAGE:-bci-verifier:0.2}" \
    bash /tests/test.sh --smoke-limit 16 >"$work/score.log" 2>&1 && \
  python3 "$TASK_ROOT/tests/campaign/check_receipt.py" "$work" \
    --tracks faced tuev ad65 openbmi longitudinal
  then
    echo 0 >"$work/exit"
  else
    echo 1 >"$work/exit"
    failures=$((failures + 1))
  fi
done
if ((failures)); then
  echo "$failures shared baseline smoke runs failed" >&2
  exit 1
fi

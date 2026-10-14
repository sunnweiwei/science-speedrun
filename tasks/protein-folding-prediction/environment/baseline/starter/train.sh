#!/usr/bin/env bash
set -euo pipefail

run_dir="${RUN_DIR:-/workspace/research/esmfold2-50m}"
index_dir="${INDEX_DIR:-${run_dir}/index}"
submission_dir="${SUBMISSION_DIR:-/workspace/submission}"
mkdir -p "${run_dir}" "${index_dir}" "${submission_dir}"

if [[ ! -s "${index_dir}/train.jsonl" ]]; then
  python -m reference.build_reference_index \
    --manifest /workspace/data/manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl \
    --output-dir "${index_dir}"
fi

publisher_pid=""
trainer_pid=""
cleanup() {
  if [[ -n "${trainer_pid}" ]]; then
    kill "${trainer_pid}" 2>/dev/null || true
    wait "${trainer_pid}" 2>/dev/null || true
  fi
  if [[ -n "${publisher_pid}" ]]; then
    kill "${publisher_pid}" 2>/dev/null || true
    wait "${publisher_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python publish_latest.py \
  --checkpoints "${run_dir}/checkpoints" \
  --submission "${submission_dir}" &
publisher_pid="$!"

train_args=(
  --train "${index_dir}/train.jsonl"
  --output-dir "${run_dir}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-100}"
  --log-interval "${LOG_INTERVAL:-10}"
  --max-tokens "${MAX_TOKENS:-384}"
  --max-atoms "${MAX_ATOMS:-4096}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  train_args+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME:-}" ]]; then
  train_args+=(--resume "${RESUME}")
fi

torchrun --standalone --nproc-per-node "${NPROC_PER_NODE:-8}" \
  -m esmfold2_50m.train "${train_args[@]}" &
trainer_pid="$!"
while kill -0 "${trainer_pid}" 2>/dev/null; do
  if ! kill -0 "${publisher_pid}" 2>/dev/null; then
    wait "${publisher_pid}"
    echo "checkpoint publisher stopped while training was active" >&2
    exit 1
  fi
  sleep 2
done
wait "${trainer_pid}"
trainer_pid=""

latest_checkpoint="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' | sort | tail -n 1)"
latest_step_text="${latest_checkpoint#step-}"
latest_step_text="${latest_step_text%.pt}"
latest_step="$((10#${latest_step_text}))"
for _ in $(seq 1 120); do
  published_step="$(python - "${submission_dir}/model_config.json" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
try:
    print(int(json.loads(path.read_text())["training_step"]))
except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    print(-1)
PY
)"
  if [[ "${published_step}" -ge "${latest_step}" ]]; then
    exit 0
  fi
  if ! kill -0 "${publisher_pid}" 2>/dev/null; then
    wait "${publisher_pid}"
  fi
  sleep 1
done
echo "publisher did not expose checkpoint step ${latest_step}" >&2
exit 1

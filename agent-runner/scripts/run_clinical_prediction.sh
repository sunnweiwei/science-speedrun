#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/zifeng/Documents/github/rsi-for-science
PYTHON312=/Users/zifeng/miniconda3/envs/py312/bin/python
VENV="$ROOT/agent-runner/.venv"
RUNNER="$VENV/bin/agent-runner"
DATA_DIR=/Users/zifeng/Documents/0-Research/EHR-dataset/clinicalbench/public
PRIVATE_LABEL_DIR=/Users/zifeng/Documents/0-Research/EHR-dataset/clinicalbench/private
PRETRAINED_DIR=/Users/zifeng/Documents/0-Research/EHR-dataset/clinicalbench/pretrained
COMPUTE_PROFILE=${COMPUTE_PROFILE:-mac}
DURATION_HOURS=${DURATION_HOURS:-24}
INDEX=${CLINICALBENCH_INDEX:-0}
MAC_IMAGE=clinicalbench-mac-codex:2.8.0-0.151.0-r3

usage() {
  echo "Usage:" >&2
  echo "  $0 start" >&2
  echo "  $0 evaluate RUN_DIR [SNAPSHOT]" >&2
}

mode=${1:-start}
case "$mode" in
  start)
    [[ $# -le 1 ]] || { usage; exit 64; }
    ;;
  evaluate)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 64; }
    ;;
  *)
    usage
    exit 64
    ;;
esac

if [[ ! -x "$RUNNER" ]]; then
  echo "agent-runner virtual environment is not initialized." >&2
  echo "Run:" >&2
  echo "  $PYTHON312 -m venv $VENV" >&2
  echo "  $VENV/bin/pip install -e $ROOT/agent-runner" >&2
  exit 1
fi

if [[ "$COMPUTE_PROFILE" == mac ]] && \
   ! docker image inspect "$MAC_IMAGE" >/dev/null 2>&1; then
  docker build \
    --tag "$MAC_IMAGE" \
    --file "$ROOT/ai4sci-tasks/clinical-prediction/local/mac/Dockerfile" \
    "$ROOT/ai4sci-tasks/clinical-prediction"
fi

case "$mode" in
  start)
    run_dir="$ROOT/runs/clinical-prediction-$(date +%Y%m%d-%H%M%S)"
    "$RUNNER" start \
      --task "$ROOT/ai4sci-tasks/clinical-prediction" \
      --run-dir "$run_dir" \
      --data-dir "$DATA_DIR" \
      --data-env-var CLINICALBENCH_DATA_DIR \
      --compute-profile "$COMPUTE_PROFILE" \
      --agent codex \
      --model openai/gpt-5.6-sol \
      --auth-file /Users/zifeng/.codex/auth.json \
      --env CLINICALBENCH_INDEX="$INDEX" \
      --env CLINICALBENCH_BACKEND_POLICY_ACK=approved \
      --env CLINICALBENCH_PRETRAINED_DIR="$PRETRAINED_DIR" \
      --network allowlist \
      --allow-host api.openai.com \
      --allow-host auth.openai.com \
      --allow-host chatgpt.com \
      --duration-hours "$DURATION_HOURS"
    echo "Run directory: $run_dir"
    echo "Log: $run_dir/runner.log"
    ;;
  evaluate)
    run_dir=$2
    snapshot=${3:-latest}
    output_dir=${EVALUATION_OUTPUT_DIR:-"$run_dir/evaluations/manual-$(date +%Y%m%d-%H%M%S)"}
    "$RUNNER" evaluate "$run_dir" \
      --snapshot "$snapshot" \
      --output-dir "$output_dir" \
      --data-dir "$DATA_DIR" \
      --gpus 0 \
      --verifier-image "$MAC_IMAGE" \
      --env CLINICALBENCH_INDEX="$INDEX" \
      --env CLINICALBENCH_BACKEND_POLICY_ACK=approved \
      --env CLINICALBENCH_PRETRAINED_DIR="$PRETRAINED_DIR" \
      --env CLINICALBENCH_PRIVATE_LABEL_DIR="$PRIVATE_LABEL_DIR"
    echo "Evaluation: $output_dir/evaluation.json"
    ;;
esac

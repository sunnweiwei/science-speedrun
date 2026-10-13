#!/usr/bin/env bash
set -euo pipefail

ROOT=${RSI_FOR_SCIENCE_ROOT:-/Users/zifeng/Documents/github/rsi-for-science}
PYTHON312=${PYTHON312:-/Users/zifeng/miniconda3/envs/py312/bin/python}
VENV=${AGENT_RUNNER_VENV:-"$ROOT/agent-runner/.venv"}
RUNNER="$VENV/bin/agent-runner"
TASK_DIR="$ROOT/ai4sci-tasks/clinical-trial-modeling"

DATA_DIR=${CLINICAL_TRIAL_MODELING_DATA_DIR:-/Users/zifeng/Documents/0-Research/clinical-trial-modeling/harbor-data-v1}
PRIVATE_LABEL_DIR=${CLINICAL_TRIAL_MODELING_PRIVATE_LABEL_DIR:-"$DATA_DIR/private"}
DOWNLOAD_CACHE_DIR=${CLINICAL_TRIAL_MODELING_DOWNLOAD_CACHE_DIR:-/Users/zifeng/Documents/0-Research/clinical-trial-modeling/download-cache}
RUNS_DIR=${CLINICAL_TRIAL_MODELING_RUNS_DIR:-"$ROOT/runs"}

COMPUTE_PROFILE=${COMPUTE_PROFILE:-mac}
MAC_IMAGE=${CLINICAL_TRIAL_MODELING_MAC_IMAGE:-clinical-trial-modeling-mac-codex:2.8.0-0.151.0-r4}
MODEL=${CODEX_MODEL:-openai/gpt-5.6-sol}
AUTH_FILE=${CODEX_AUTH_FILE:-/Users/zifeng/.codex/auth.json}
DURATION_HOURS=${DURATION_HOURS:-1} # debug
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-300}

usage() {
  cat >&2 <<EOF
Usage:
  $0 materialize
  $0 build-mac
  $0 start [RUN_DIR]
  $0 evaluate RUN_DIR [SNAPSHOT]
  $0 status RUN_DIR
  $0 snapshot RUN_DIR
  $0 stop RUN_DIR
  $0 list

Common overrides:
  CLINICAL_TRIAL_MODELING_DATA_DIR=$DATA_DIR
  CLINICAL_TRIAL_MODELING_PRIVATE_LABEL_DIR=$PRIVATE_LABEL_DIR
  CLINICAL_TRIAL_MODELING_DOWNLOAD_CACHE_DIR=$DOWNLOAD_CACHE_DIR
  CLINICAL_TRIAL_MODELING_CT_SOURCE_DIR=/path/to/CT_Open
  CLINICAL_TRIAL_MODELING_TRIALBENCH_ARCHIVE=/path/to/all_task.zip
  COMPUTE_PROFILE=mac|default
  DURATION_HOURS=$DURATION_HOURS
  CODEX_MODEL=$MODEL
  CODEX_AUTH_FILE=$AUTH_FILE
EOF
}

die() {
  echo "run_clinical_trial_modeling.sh: $*" >&2
  exit 1
}

require_runner() {
  if [[ ! -x "$RUNNER" ]]; then
    cat >&2 <<EOF
agent-runner virtual environment is not initialized. Run:
  $PYTHON312 -m venv "$VENV"
  "$VENV/bin/pip" install -e "$ROOT/agent-runner"
EOF
    exit 1
  fi
}

build_mac_image() {
  if docker image inspect "$MAC_IMAGE" >/dev/null 2>&1; then
    return
  fi
  echo "Building Mac CPU image: $MAC_IMAGE"
  docker build \
    --tag "$MAC_IMAGE" \
    --file "$TASK_DIR/local/mac/Dockerfile" \
    "$TASK_DIR"
}

require_materialized_data() {
  [[ -s "$DATA_DIR/public/development/manifest.json" ]] || {
    echo "Materialized development data is missing:" >&2
    echo "  $DATA_DIR/public/development/manifest.json" >&2
    echo "Run '$0 materialize' first." >&2
    exit 1
  }
}

mode=${1:-start}
case "$mode" in
  materialize|build-mac|list)
    [[ $# -eq 1 || $# -eq 0 ]] || { usage; exit 64; }
    ;;
  start)
    [[ $# -le 2 ]] || { usage; exit 64; }
    ;;
  evaluate)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 64; }
    ;;
  status|snapshot|stop)
    [[ $# -eq 2 ]] || { usage; exit 64; }
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 64
    ;;
esac

require_runner

case "$mode" in
  materialize)
    materialize_args=(
      materialize
      --task "$TASK_DIR"
      --data-dir "$DATA_DIR"
      --env "CLINICAL_TRIAL_MODELING_DOWNLOAD_CACHE_DIR=$DOWNLOAD_CACHE_DIR"
      --env "CLINICAL_TRIAL_MODELING_PRIVATE_LABEL_DIR=$PRIVATE_LABEL_DIR"
    )
    if [[ -n ${CLINICAL_TRIAL_MODELING_CT_SOURCE_DIR:-} ]]; then
      materialize_args+=(--env "CLINICAL_TRIAL_MODELING_CT_SOURCE_DIR=$CLINICAL_TRIAL_MODELING_CT_SOURCE_DIR")
    fi
    if [[ -n ${CLINICAL_TRIAL_MODELING_TRIALBENCH_ARCHIVE:-} ]]; then
      materialize_args+=(--env "CLINICAL_TRIAL_MODELING_TRIALBENCH_ARCHIVE=$CLINICAL_TRIAL_MODELING_TRIALBENCH_ARCHIVE")
    fi
    "$RUNNER" "${materialize_args[@]}"
    echo "Public development data: $DATA_DIR/public/development"
    echo "Public verifier data:    $DATA_DIR/public/verifier"
    echo "Private labels:          $PRIVATE_LABEL_DIR"
    ;;

  build-mac)
    build_mac_image
    echo "Mac CPU image ready: $MAC_IMAGE"
    ;;

  start)
    require_materialized_data
    [[ -f "$AUTH_FILE" ]] || die "Codex auth file is missing: $AUTH_FILE"
    if [[ "$COMPUTE_PROFILE" == mac ]]; then
      build_mac_image
    fi
    mkdir -p "$RUNS_DIR"
    run_dir=${2:-"$RUNS_DIR/clinical-trial-modeling-$(date +%Y%m%d-%H%M%S)"}
    start_args=(
      start
      --task "$TASK_DIR"
      --run-dir "$run_dir"
      --data-dir "$DATA_DIR"
      --data-env-var CLINICAL_TRIAL_MODELING_DATA_DIR
      --compute-profile "$COMPUTE_PROFILE"
      --agent codex
      --model "$MODEL"
      --auth-file "$AUTH_FILE"
      --network allowlist
      --allow-host api.openai.com
      --allow-host auth.openai.com
      --allow-host chatgpt.com
      --duration-hours "$DURATION_HOURS"
      --snapshot-interval "$SNAPSHOT_INTERVAL"
    )
    if [[ -n ${CODEX_BIN:-} ]]; then
      start_args+=(--agent-bin "codex=$CODEX_BIN")
    fi
    if [[ -n ${CODEX_CODE_MODE_HOST_BIN:-} ]]; then
      start_args+=(--agent-bin "codex-code-mode-host=$CODEX_CODE_MODE_HOST_BIN")
    fi
    "$RUNNER" "${start_args[@]}"
    echo "Run directory: $run_dir"
    echo "Status:        $0 status '$run_dir'"
    echo "Log:           $run_dir/runner.log"
    ;;

  evaluate)
    run_dir=$2
    snapshot=${3:-latest}
    [[ -s "$DATA_DIR/public/verifier/manifest.json" ]] || \
      die "materialized verifier data is missing under $DATA_DIR"
    [[ -s "$PRIVATE_LABEL_DIR/manifest.json" ]] || \
      die "private labels are missing under $PRIVATE_LABEL_DIR"
    output_dir=${EVALUATION_OUTPUT_DIR:-"$run_dir/evaluations/manual-$(date +%Y%m%d-%H%M%S)"}
    evaluate_args=(
      evaluate "$run_dir"
      --snapshot "$snapshot"
      --output-dir "$output_dir"
      --data-dir "$DATA_DIR"
      --env "CLINICAL_TRIAL_MODELING_PRIVATE_LABEL_DIR=$PRIVATE_LABEL_DIR"
    )
    if [[ "$COMPUTE_PROFILE" == mac ]]; then
      evaluate_args+=(--verifier-image "$MAC_IMAGE")
    fi
    "$RUNNER" "${evaluate_args[@]}"
    trial_report=$(find "$output_dir/harbor" -mindepth 3 -maxdepth 3 -path '*/verifier/report.json' -type f -print -quit)
    if [[ -n "$trial_report" ]]; then
      cp "$trial_report" "$output_dir/report.json"
      echo "Detailed task report: $output_dir/report.json"
    fi
    echo "Evaluation metadata: $output_dir/evaluation.json"
    ;;

  status)
    "$RUNNER" status "$2"
    ;;

  snapshot)
    "$RUNNER" snapshot "$2"
    ;;

  stop)
    "$RUNNER" stop "$2"
    ;;

  list)
    "$RUNNER" list --runs-dir "$RUNS_DIR"
    ;;
esac

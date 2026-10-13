#!/usr/bin/env bash
set -euo pipefail

ROOT=${RSI_FOR_SCIENCE_ROOT:-/Users/zifeng/Documents/github/rsi-for-science}
PYTHON312=${PYTHON312:-/Users/zifeng/miniconda3/envs/py312/bin/python}
VENV=${AGENT_RUNNER_VENV:-"$ROOT/agent-runner/.venv"}
MATERIALIZE_PYTHON=${BIOMED_MATERIALIZE_PYTHON:-"$VENV/bin/python"}
RUNNER="$VENV/bin/agent-runner"
TASK_DIR="$ROOT/ai4sci-tasks/biomedicine-deep-research"

TASK_DATA_VERSION=1.1.0
DATA_DIR=${BIOMED_DEEP_RESEARCH_DATA_DIR:-/Users/zifeng/Documents/0-Research/biomedicine-deep-research/harbor-data-v1}
DOWNLOAD_CACHE_DIR=${BIOMED_DEEP_RESEARCH_DOWNLOAD_CACHE_DIR:-/Users/zifeng/Documents/0-Research/biomedicine-deep-research/download-cache}
RUNS_DIR=${BIOMED_DEEP_RESEARCH_RUNS_DIR:-"$ROOT/runs"}

COMPUTE_PROFILE=${COMPUTE_PROFILE:-gemini-3-6-flash}
MAC_IMAGE=${BIOMED_DEEP_RESEARCH_MAC_IMAGE:-biomedicine-deep-research-mac-codex:0.151.0-r1}
MODEL=${CODEX_MODEL:-openai/gpt-5.6-sol}
AUTH_FILE=${CODEX_AUTH_FILE:-/Users/zifeng/.codex/auth.json}
DURATION_HOURS=${DURATION_HOURS:-1} # debug
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-300}
GATEWAY_ONLINE=${BIOMED_GATEWAY_ONLINE:-1}
GATEWAY_ADMIN_TOKEN=${BIOMED_GATEWAY_ADMIN_TOKEN:-}

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
  BIOMED_DEEP_RESEARCH_DATA_DIR=$DATA_DIR
  BIOMED_DEEP_RESEARCH_DOWNLOAD_CACHE_DIR=$DOWNLOAD_CACHE_DIR
  BIOMED_DEEPEVIDENCE_DIR=/path/to/DeepEvidence
  BIOMED_BIODSA_DIR=/path/to/BioDSA
  BIOMED_CLINICAL_TOOL_LEARNING_DIR=/path/to/Clinical-Tool-Learning
  BIOMED_MATERIALIZE_PYTHON=$MATERIALIZE_PYTHON
  BIOMED_MATERIALIZE_OFFLINE=0|1
  BIOMED_DEEP_RESEARCH_MAC_IMAGE=$MAC_IMAGE
  COMPUTE_PROFILE=<profile name from compute-profiles.toml>
  EVALUATION_COMPUTE_PROFILE=<optional evaluate override>
  BIOMED_GATEWAY_ONLINE=0|1
  CODEX_MODEL=$MODEL
  CODEX_AUTH_FILE=$AUTH_FILE
  DURATION_HOURS=$DURATION_HOURS

Online gateway credentials for the selected profile:
  OPENAI_API_KEY
  AZURE_API_KEY + AZURE_API_BASE + AZURE_API_VERSION
  GEMINI_API_KEY
  ANTHROPIC_API_KEY
EOF
}

die() {
  echo "run_biomedicine_deep_research.sh: $*" >&2
  exit 1
}

require_runner() {
  if [[ ! -x "$RUNNER" ]]; then
    cat >&2 <<EOF
agent-runner virtual environment is not initialized. Run:
  $PYTHON312 -m venv "$VENV"
  "$VENV/bin/pip" install -e '$ROOT/agent-runner[test,biomedicine]'
EOF
    exit 1
  fi
}

require_materialized_data() {
  [[ -s "$DATA_DIR/public/development/manifest.json" ]] || \
    die "development data is missing; run '$0 materialize' first"
  [[ -s "$DATA_DIR/public/verifier/manifest.json" ]] || \
    die "verifier data is missing; run '$0 materialize' first"
  [[ -s "$DATA_DIR/private/manifest.json" ]] || \
    die "private labels are missing; run '$0 materialize' first"
  [[ -s "$DATA_DIR/public/reference/source-allowlist.json" ]] || \
    die "source allowlist is missing; task v1.1.0 must be materialized into a new directory"
  [[ -s "$DATA_DIR/public/reference/riskcalcs.json" ]] || \
    die "RiskCalcs is missing; task v1.1.0 must be materialized into a new directory"
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

ensure_gateway_token() {
  if [[ -z "$GATEWAY_ADMIN_TOKEN" ]]; then
    if command -v openssl >/dev/null 2>&1; then
      GATEWAY_ADMIN_TOKEN=$(openssl rand -hex 32)
    else
      GATEWAY_ADMIN_TOKEN="local-$(date +%s)-$$"
    fi
  fi
}

append_if_set() {
  local name=$1
  if [[ -n ${!name:-} ]]; then
    gateway_env_args+=(--env "$name=${!name}")
  fi
}

prepare_gateway_env() {
  local profile=$1
  local provider=${2:-}
  if [[ -z "$provider" ]]; then
    provider=$("$VENV/bin/python" -c \
      'import sys; from pathlib import Path; from agent_runner.task import resolve_compute_profile; print(resolve_compute_profile(Path(sys.argv[1]), sys.argv[2])["harness_llm_provider"])' \
      "$TASK_DIR" "$profile") || die "cannot resolve compute profile: $profile"
  fi
  gateway_env_args=()
  ensure_gateway_token
  gateway_env_args+=(
    --env "BIOMED_GATEWAY_ADMIN_TOKEN=$GATEWAY_ADMIN_TOKEN"
    --env "BIOMED_GATEWAY_ONLINE=$GATEWAY_ONLINE"
  )

  case "$provider" in
    openai)
      [[ "$GATEWAY_ONLINE" != 1 || -n ${OPENAI_API_KEY:-} ]] || \
        die "OPENAI_API_KEY is required for online profile $profile"
      ;;
    azure)
      if [[ "$GATEWAY_ONLINE" == 1 ]]; then
        [[ -n ${AZURE_API_KEY:-} ]] || die "AZURE_API_KEY is required for $profile"
        [[ -n ${AZURE_API_BASE:-} ]] || die "AZURE_API_BASE is required for $profile"
        [[ -n ${AZURE_API_VERSION:-} ]] || die "AZURE_API_VERSION is required for $profile"
      fi
      ;;
    gemini)
      [[ "$GATEWAY_ONLINE" != 1 || -n ${GEMINI_API_KEY:-} ]] || \
        die "GEMINI_API_KEY is required for online profile $profile"
      ;;
    claude|anthropic)
      [[ "$GATEWAY_ONLINE" != 1 || -n ${ANTHROPIC_API_KEY:-} ]] || \
        die "ANTHROPIC_API_KEY is required for online profile $profile"
      ;;
    *)
      die "unsupported harness LLM provider '$provider' in profile $profile"
      ;;
  esac

  append_if_set OPENAI_API_KEY
  append_if_set OPENAI_API_BASE
  append_if_set OPENAI_BASE_URL
  append_if_set AZURE_API_KEY
  append_if_set AZURE_API_BASE
  append_if_set AZURE_API_VERSION
  append_if_set GEMINI_API_KEY
  append_if_set ANTHROPIC_API_KEY
}

mode=${1:-start}
case "$mode" in
  materialize|build-mac|list)
    [[ $# -le 1 ]] || { usage; exit 64; }
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
    [[ -x "$MATERIALIZE_PYTHON" ]] || die "materialization Python is missing: $MATERIALIZE_PYTHON"
    "$MATERIALIZE_PYTHON" -c 'import pyarrow' >/dev/null 2>&1 || die \
      "pyarrow is missing; install it for $MATERIALIZE_PYTHON or set BIOMED_MATERIALIZE_PYTHON"
    materialize_args=(--output "$DATA_DIR" --cache "$DOWNLOAD_CACHE_DIR" --overwrite)
    if [[ ${BIOMED_MATERIALIZE_OFFLINE:-0} == 1 ]]; then
      materialize_args+=(--offline)
    fi
    "$MATERIALIZE_PYTHON" "$TASK_DIR/environment/data/materialize.py" "${materialize_args[@]}"
    echo "Development data: $DATA_DIR/public/development"
    echo "Verifier data:    $DATA_DIR/public/verifier"
    echo "Private labels:   $DATA_DIR/private"
    echo "Reference data:   $DATA_DIR/public/reference"
    ;;

  build-mac)
    build_mac_image
    echo "Mac CPU image ready: $MAC_IMAGE"
    ;;

  start)
    require_materialized_data
    [[ -f "$AUTH_FILE" ]] || die "Codex auth file is missing: $AUTH_FILE"
    prepare_gateway_env "$COMPUTE_PROFILE"
    build_mac_image
    mkdir -p "$RUNS_DIR"
    run_dir=${2:-"$RUNS_DIR/biomedicine-deep-research-$(date +%Y%m%d-%H%M%S)"}
    start_args=(
      start
      --task "$TASK_DIR"
      --run-dir "$run_dir"
      --data-dir "$DATA_DIR"
      --data-env-var BIOMED_DEEP_RESEARCH_DATA_DIR
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
      "${gateway_env_args[@]}"
    )
    if [[ -n ${CODEX_BIN:-} ]]; then
      start_args+=(--agent-bin "codex=$CODEX_BIN")
    fi
    if [[ -n ${CODEX_CODE_MODE_HOST_BIN:-} ]]; then
      start_args+=(--agent-bin "codex-code-mode-host=$CODEX_CODE_MODE_HOST_BIN")
    fi
    "$RUNNER" "${start_args[@]}"
    echo "Run directory: $run_dir"
    echo "Core LLM:      $COMPUTE_PROFILE"
    echo "Status:        $0 status '$run_dir'"
    echo "Log:           $run_dir/runner.log"
    ;;

  evaluate)
    require_materialized_data
    run_dir=$2
    snapshot=${3:-latest}
    [[ -s "$run_dir/run.json" ]] || die "run metadata is missing: $run_dir/run.json"
    if [[ -n ${EVALUATION_COMPUTE_PROFILE:-} ]]; then
      evaluation_profile=$EVALUATION_COMPUTE_PROFILE
      evaluation_provider=""
      evaluation_profile_args=(--compute-profile "$evaluation_profile")
    else
      evaluation_profile=$("$VENV/bin/python" -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["compute_profile"])' \
        "$run_dir/run.json")
      evaluation_provider=$("$VENV/bin/python" -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["harness_llm_provider"])' \
        "$run_dir/run.json")
      evaluation_profile_args=()
    fi
    prepare_gateway_env "$evaluation_profile" "$evaluation_provider"
    output_dir=${EVALUATION_OUTPUT_DIR:-"$run_dir/evaluations/manual-$(date +%Y%m%d-%H%M%S)"}
    "$RUNNER" evaluate "$run_dir" \
      --snapshot "$snapshot" \
      --output-dir "$output_dir" \
      --data-dir "$DATA_DIR" \
      "${evaluation_profile_args[@]}" \
      "${gateway_env_args[@]}"
    echo "Core LLM:   $evaluation_profile"
    echo "Evaluation: $output_dir/evaluation.json"
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

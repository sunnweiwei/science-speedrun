#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
environment_dir="$(cd -- "$script_dir/.." && pwd)"
task_root="$(cd -- "$environment_dir/.." && pwd)"

image="${TARGET_BASED_DRUG_SCREENING_MATERIALIZER_IMAGE:-${DRUGCLIP_MATERIALIZER_IMAGE:-target-based-drug-screening-materializer:0.1.0}}"
workers="${TARGET_BASED_DRUG_SCREENING_WORKERS:-${DRUGCLIP_WORKERS:-24}}"
receptor_cache_size="${TARGET_BASED_DRUG_SCREENING_RECEPTOR_CACHE_SIZE:-8}"
task_chunk_size="${TARGET_BASED_DRUG_SCREENING_TASK_CHUNK_SIZE:-16}"
minimum_free_gib="${TARGET_BASED_DRUG_SCREENING_MIN_FREE_GIB:-${DRUGCLIP_MIN_FREE_GIB:-90}}"
max_retries="${TARGET_BASED_DRUG_SCREENING_MAX_RETRIES:-${DRUGCLIP_MAX_RETRIES:-10}}"
retry_delay="${TARGET_BASED_DRUG_SCREENING_RETRY_DELAY_SECONDS:-${DRUGCLIP_RETRY_DELAY_SECONDS:-30}}"

usage() {
  cat <<'EOF'
Usage: bash environment/data/prepare.sh [MODE] [TARGET_DIRECTORY]

Modes:
  check  Check Docker and disk space; do not download or build anything.
  smoke  Build the image and run the complete pipeline on synthetic fixtures.
  full   Run an isolated smoke test, then download, prepare, and validate all
         data. This is the default when MODE is omitted.

TARGET_DIRECTORY defaults to TARGET_BASED_DRUG_SCREENING_DATA_DIR when set,
then the legacy DRUGCLIP_DATA_DIR, otherwise to <task>/data. Downloads and
completed preprocessing caches are reused after an interruption. Full mode
downloads the compact Hugging Face release by default. Set
TARGET_BASED_DRUG_SCREENING_DATA_SOURCE=upstream to run the original-source
pipeline. At least 90 GiB free is conservatively enforced for staging and
test archive extraction.

Examples:
  bash environment/data/prepare.sh
  bash environment/data/prepare.sh check /mnt/d/datasets/target-based-drug-screening
  TARGET_BASED_DRUG_SCREENING_DATA_DIR=/data/target-screening bash environment/data/prepare.sh full
EOF
}

mode="full"
if [[ $# -gt 0 ]]; then
  case "$1" in
    check|smoke|full) mode="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown mode: $1" >&2; usage >&2; exit 2 ;;
  esac
fi
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi

target="${1:-${TARGET_BASED_DRUG_SCREENING_DATA_DIR:-${DRUGCLIP_DATA_DIR:-$task_root/data}}}"
mkdir -p -- "$target"
target="$(cd -- "$target" && pwd)"

command -v docker >/dev/null 2>&1 || {
  echo "ERROR: Docker is not installed or is not on PATH." >&2; exit 2;
}
docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker cannot connect to its daemon." >&2
  echo "Start Docker Desktop and enable WSL integration, then retry." >&2
  exit 2
}

for value_name in minimum_free_gib max_retries retry_delay workers receptor_cache_size task_chunk_size; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( max_retries < 1 || workers < 1 || receptor_cache_size < 1 || task_chunk_size < 1 )); then
  echo "ERROR: retry, worker, receptor-cache, and task-chunk counts must be at least 1." >&2
  exit 2
fi

available_kib="$(df -Pk -- "$target" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not determine free space for $target" >&2; exit 2
fi
available_gib=$((available_kib / 1024 / 1024))
echo "Mode: $mode"
echo "Target: $target"
echo "Free space: ${available_gib} GiB"
echo "Workers: $workers"

if [[ "$mode" == "full" ]] && (( available_gib < minimum_free_gib )); then
  echo "ERROR: full mode requires at least ${minimum_free_gib} GiB free." >&2
  echo "Set TARGET_BASED_DRUG_SCREENING_MIN_FREE_GIB only if storage requirements are handled externally." >&2
  exit 2
fi
if [[ "$mode" == "check" ]]; then
  echo "PASS: Docker and disk preflight succeeded; no data was downloaded."
  exit 0
fi

echo "Building materializer image (Docker will reuse its build cache)..."
build_log="$(mktemp "${TMPDIR:-/tmp}/target-screening-docker-build.XXXXXX")"
anonymous_docker_config=""
cleanup_temporary_files() {
  rm -f -- "$build_log"
  if [[ -n "$anonymous_docker_config" ]]; then rm -rf -- "$anonymous_docker_config"; fi
}
trap cleanup_temporary_files EXIT

if ! docker build -f "$environment_dir/Materialize.Dockerfile" -t "$image" \
    "$environment_dir" 2>&1 | tee "$build_log"; then
  if ! grep -qi "error getting credentials" "$build_log"; then
    echo "ERROR: Docker image build failed." >&2; exit 1
  fi
  anonymous_docker_config="$(mktemp -d "${TMPDIR:-/tmp}/target-screening-docker-config.XXXXXX")"
  printf '{"auths":{}}\n' > "$anonymous_docker_config/config.json"
  export DOCKER_CONFIG="$anonymous_docker_config"
  echo "Docker credential helper failed; retrying public image build anonymously..." >&2
  docker build -f "$environment_dir/Materialize.Dockerfile" -t "$image" "$environment_dir"
fi

run_materializer() {
  local host_root="$1"; shift
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e TARGET_BASED_DRUG_SCREENING_WORKERS="$workers" \
    -e TARGET_BASED_DRUG_SCREENING_RECEPTOR_CACHE_SIZE="$receptor_cache_size" \
    -e TARGET_BASED_DRUG_SCREENING_TASK_CHUNK_SIZE="$task_chunk_size" \
    -e TARGET_BASED_DRUG_SCREENING_DATA_SOURCE="${TARGET_BASED_DRUG_SCREENING_DATA_SOURCE:-huggingface}" \
    -e TARGET_BASED_DRUG_SCREENING_HF_REPO \
    -e TARGET_BASED_DRUG_SCREENING_HF_REVISION \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -v "$host_root:/data" \
    "$image" python3 -m drugclip_data.cli "$@" --data-root /data
}

write_status() {
  local state="$1" detail="${2:-}" status_file="$target/materialize.status"
  printf 'state=%s\ndetail=%s\npid=%s\nupdated_at=%s\n' \
    "$state" "$detail" "$$" "$(date --iso-8601=seconds)" > "$status_file.tmp"
  mv "$status_file.tmp" "$status_file"
}

smoke_root="$target/.smoke"
mkdir -p -- "$smoke_root"
if [[ ! -s "$smoke_root/.complete.json" ]]; then
  echo "Running isolated end-to-end smoke test (synthetic data only)..."
  run_materializer "$smoke_root" all --smoke --overwrite
fi
run_materializer "$smoke_root" validate --smoke
if [[ "$mode" == "smoke" ]]; then
  echo "PASS: synthetic end-to-end smoke test completed at $smoke_root"; exit 0
fi

attempt=1
while true; do
  write_status "MATERIALIZING" "attempt=$attempt"
  set +e
  run_materializer "$target" release
  exit_code=$?
  set -e
  if (( exit_code == 0 )); then break; fi
  if (( attempt >= max_retries )); then
    write_status "FAILED" "attempts exhausted: $attempt; exit_code=$exit_code"
    echo "ERROR: materialization failed after $attempt attempt(s)." >&2
    echo "Rerun the same command to resume verified downloads and cached preprocessing." >&2
    exit "$exit_code"
  fi
  attempt=$((attempt + 1))
  write_status "RETRY_WAIT" "next_attempt=$attempt delay_seconds=$retry_delay"
  echo "Attempt failed; retrying resumably in ${retry_delay}s (attempt $attempt/$max_retries)..." >&2
  sleep "$retry_delay"
done

if [[ ! -s "$target/.complete.json" ]]; then
  write_status "FAILED" "materializer exited successfully without a completion receipt"
  echo "ERROR: materializer did not create $target/.complete.json" >&2; exit 1
fi
write_status "VALIDATING" "materializer completed"
run_materializer "$target" validate
write_status "COMPLETE" "validation passed"
echo "PASS: target-based drug-screening data are ready at $target"


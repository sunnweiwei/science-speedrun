#!/usr/bin/env bash
# Medium single-GPU learning-curve pilot on fixed LMDB prefixes. This is not
# the formal full-data 8/4-GPU recipe. Resume by rerunning the same command.
set -euo pipefail

DATA_ROOT="$(realpath "${1:?data root required}")"
RUN_ROOT="$(realpath -m "${2:?pilot output directory required}")"
IMAGE="${DRUGCLIP_SMOKE_IMAGE:-drugclip-paper-smoke:local}"
CODE_ROOT="$(cd -P "$(dirname "$0")" && pwd)"
SCORE_SCRIPT="$(realpath "$CODE_ROOT/../../evaluation/score.py")"
mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/started_utc.txt" ]]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/started_utc.txt"
fi

run_train() {
  docker run --rm --gpus all --shm-size=4g \
    -v "$CODE_ROOT:/code:ro" -v "$DATA_ROOT:/data:ro" \
    -v "$RUN_ROOT:/run" -w /code "$IMAGE" python train.py "$@"
}

COMMON=(--input-root /data --workers 0 --device cuda --batch-size 8
  --validation-batch-size 32 --profsa-epochs 3 --finetune-epochs 6
  --max-pretrain-records 512 --max-train-records 2048
  --max-validation-records 256)

if [[ ! -f "$RUN_ROOT/profsa-work/profsa_complete.pt" ]]; then
  RESUME=()
  if [[ -f "$RUN_ROOT/profsa-work/last.pt" ]]; then
    RESUME=(--resume /run/profsa-work/last.pt)
  fi
  run_train --stage profsa "${COMMON[@]}" \
    --work-dir /run/profsa-work --output-dir /run/unused-profsa-model \
    "${RESUME[@]}" 2>&1 | tee -a "$RUN_ROOT/profsa.log"
fi
test -f "$RUN_ROOT/profsa-work/profsa_complete.pt"

if [[ ! -f "$RUN_ROOT/finetune-work/training_summary.json" ]]; then
  RESUME=()
  if [[ -f "$RUN_ROOT/finetune-work/last.pt" ]]; then
    RESUME=(--resume /run/finetune-work/last.pt)
  fi
  run_train --stage finetune "${COMMON[@]}" \
    --work-dir /run/finetune-work --output-dir /run/model \
    --profsa-checkpoint /run/profsa-work/profsa_complete.pt \
    "${RESUME[@]}" 2>&1 | tee -a "$RUN_ROOT/finetune.log"
fi
test -f "$RUN_ROOT/model/model.pt"

# Only a copied package and one complete test target are visible to inference.
if [[ ! -f "$RUN_ROOT/predictions/metrics.json" ]]; then
  TARGET="$(find "$DATA_ROOT/test/dude" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | head -1)"
  if [[ ! -d "$RUN_ROOT/standalone-model" ]]; then
    cp -a "$RUN_ROOT/model" "$RUN_ROOT/standalone-model"
  fi
  if [[ ! -f "$RUN_ROOT/predictions/scores.csv" ]]; then
    rm -rf "$RUN_ROOT/predictions"
    mkdir -p "$RUN_ROOT/predictions"
    docker run --rm --gpus all --shm-size=4g \
      -v "$RUN_ROOT/standalone-model:/package:ro" \
      -v "$DATA_ROOT/test/dude/$TARGET:/eval-data/test/dude/$TARGET:ro" \
      -v "$RUN_ROOT/predictions:/predictions" -w /package "$IMAGE" \
      python /package/predict.py /eval-data /predictions --benchmark dude \
      --max-targets 1 --batch-size 8 --workers 0 --device cuda \
      2>&1 | tee "$RUN_ROOT/inference.log"
  fi
  docker run --rm \
    -v "$DATA_ROOT/test/dude/$TARGET:/eval-data/test/dude/$TARGET:ro" \
    -v "$RUN_ROOT/predictions:/predictions" \
    -v "$(dirname "$SCORE_SCRIPT"):/evaluation:ro" "$IMAGE" \
    python /evaluation/score.py /eval-data /predictions/scores.csv \
    /predictions/metrics.json 2>&1 | tee "$RUN_ROOT/scoring.log"
fi

sha256sum "$RUN_ROOT/standalone-model/model.pt" > "$RUN_ROOT/model.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/finished_utc.txt"
BASELINE_ARGS=()
if [[ -f "$DATA_ROOT/reference-smoke-v3/smoke_report.json" ]]; then
  BASELINE_ARGS=(--baseline-report /data/reference-smoke-v3/smoke_report.json)
fi
docker run --rm -v "$CODE_ROOT:/code:ro" -v "$DATA_ROOT:/data:ro" \
  -v "$RUN_ROOT:/run" -w /code "$IMAGE" \
  python summarize_single_gpu_pilot.py /run "${BASELINE_ARGS[@]}" \
  > "$RUN_ROOT/report.log"


#!/usr/bin/env bash
# Run the full 95M architecture against small real train/dev slices, then one
# complete DUD-E target. Requires Docker with NVIDIA GPU support.
set -euo pipefail

DATA_ROOT="$(realpath "${1:?data root required}")"
SMOKE_ROOT="$(realpath -m "${2:?new smoke output directory required}")"
IMAGE="${DRUGCLIP_SMOKE_IMAGE:-drugclip-paper-smoke:local}"
CODE_ROOT="$(cd -P "$(dirname "$0")" && pwd)"
SCORE_SCRIPT="$(realpath "$CODE_ROOT/../../evaluation/score.py")"
if [[ -e "$SMOKE_ROOT" ]] && [[ -n "$(ls -A "$SMOKE_ROOT")" ]]; then
  echo "Smoke directory must start empty: $SMOKE_ROOT" >&2
  exit 1
fi
mkdir -p "$SMOKE_ROOT"
TARGET="$(find "$DATA_ROOT/test/dude" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | head -1)"
echo "GPU smoke: target=$TARGET, image=$IMAGE, output=$SMOKE_ROOT"

run_train() {
  docker run --rm --gpus all --shm-size=4g \
    -v "$CODE_ROOT:/code:ro" -v "$DATA_ROOT:/data:ro" \
    -v "$SMOKE_ROOT:/smoke" -w /code "$IMAGE" \
    python train.py "$@"
}

COMMON=(--input-root /data --workers 0 --device cuda --batch-size 2
  --validation-batch-size 4 --profsa-epochs 2 --finetune-epochs 2
  --max-pretrain-records 8 --max-train-records 8 --max-validation-records 8)

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$SMOKE_ROOT/started_utc.txt"
run_train --stage profsa "${COMMON[@]}" --work-dir /smoke/profsa-work \
  --output-dir /smoke/unused-profsa-model 2>&1 | tee "$SMOKE_ROOT/profsa.log"
test -f "$SMOKE_ROOT/profsa-work/profsa_complete.pt"

FINETUNE=(--stage finetune "${COMMON[@]}" --work-dir /smoke/finetune-work
  --output-dir /smoke/model
  --profsa-checkpoint /smoke/profsa-work/profsa_complete.pt
  --max-seconds 0.001)
run_train "${FINETUNE[@]}" 2>&1 | tee "$SMOKE_ROOT/finetune-stop.log"
cp "$SMOKE_ROOT/finetune-work/last.pt" \
  "$SMOKE_ROOT/finetune_epoch1_last.pt"
run_train "${FINETUNE[@]}" --resume /smoke/finetune-work/last.pt \
  2>&1 | tee "$SMOKE_ROOT/finetune-resume.log"

# Expose only the copied package and one test target to the inference process;
# it cannot read the training work directory or pretrained source files.
cp -a "$SMOKE_ROOT/model" "$SMOKE_ROOT/standalone-model"
mkdir -p "$SMOKE_ROOT/predictions"
docker run --rm --gpus all --shm-size=4g \
  -v "$SMOKE_ROOT/standalone-model:/package:ro" \
  -v "$DATA_ROOT/test/dude/$TARGET:/eval-data/test/dude/$TARGET:ro" \
  -v "$SMOKE_ROOT/predictions:/predictions" -w /package "$IMAGE" \
  python /package/predict.py /eval-data /predictions --benchmark dude \
  --max-targets 1 --batch-size 8 --workers 0 --device cuda \
  2>&1 | tee "$SMOKE_ROOT/inference.log"
docker run --rm \
  -v "$DATA_ROOT/test/dude/$TARGET:/eval-data/test/dude/$TARGET:ro" \
  -v "$SMOKE_ROOT/predictions:/predictions" \
  -v "$(dirname "$SCORE_SCRIPT"):/evaluation:ro" "$IMAGE" \
  python /evaluation/score.py /eval-data /predictions/scores.csv \
  /predictions/metrics.json 2>&1 | tee "$SMOKE_ROOT/scoring.log"
sha256sum "$SMOKE_ROOT/standalone-model/model.pt" \
  > "$SMOKE_ROOT/model.sha256"
docker run --rm -v "$CODE_ROOT:/code:ro" -v "$DATA_ROOT:/data:ro" \
  -v "$SMOKE_ROOT:/smoke" -w /code "$IMAGE" \
  python audit_single_gpu_smoke.py /data /smoke \
  > "$SMOKE_ROOT/audit.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$SMOKE_ROOT/finished_utc.txt"


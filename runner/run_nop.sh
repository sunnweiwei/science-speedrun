#!/usr/bin/env bash
# Harbor nop trial (empty submission): checks that the task builds, the verifier runs and
# reward.json parses (an empty package must score 0 with invalid_submission, not crash).
#   run_nop.sh NAME TASK GPU_ID
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
name=$1; task=$2; gpu=$3; shift 3
mkdir -p "$EVAL_ROOT/trials-nop" "$EVAL_ROOT/research/$name"
$DOCKER run --rm --name "nop-$name" $(runner_mounts) -w "$EVAL_ROOT" \
  -e PYTHONPATH="$HARNESS_DIR" -e HARBOR_GPU_DEVICE_IDS="$gpu" -e HARBOR_GPU_CAP=1 -e HARBOR_SKIP_VERIFIER=0 \
  -e TASK_RESEARCH_DIR="$EVAL_ROOT/research/$name" ${DATA_ENV_FILE:+--env-file $DATA_ENV_FILE} \
  "$RUNNER_IMAGE" harbor trial start -p "$TASKS_DIR/$task" -a nop \
  --env harbor_ext.local_gpu_docker:LocalGpuDockerEnvironment --override-gpus 1 \
  --agent-timeout 120 --trial-name "$name" --trials-dir "$EVAL_ROOT/trials-nop" "$@"

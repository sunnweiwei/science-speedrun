#!/usr/bin/env bash
# Score a saved package with the task's own verifier resources (task.toml GPUs, CPUs, memory,
# timeout): a copy agent publishes PACKAGE_DIR, then Harbor runs the separate verifier.
# Serialized with a lock so full-resource verifications never overlap.
#   run_verify.sh NAME TASK PACKAGE_DIR
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
name=$1; task=$2; pkg=$(cd "$3" && pwd)
data_env="${DATA_ENV_FILE:-$ENVS_DIR/$task.env}"
mkdir -p "$EVAL_ROOT/trials-verify" "$EVAL_ROOT/research/$name"
exec 9>"$EVAL_ROOT/.verify.lock"; flock 9
$DOCKER run --rm --name "verify-$name" $(runner_mounts) -v "$pkg:$pkg" -w "$EVAL_ROOT" \
  -e PYTHONPATH="$HARNESS_DIR" -e HARBOR_GPU_DEVICE_IDS="${GPU_IDS:-}" \
  -e HARBOR_SKIP_VERIFIER=0 -e HARBOR_COPY_PACKAGE="$pkg" \
  -e TASK_RESEARCH_DIR="$EVAL_ROOT/research/$name" --env-file "$data_env" \
  "$RUNNER_IMAGE" harbor trial start -p "$TASKS_DIR/$task" -a harbor_ext.copy_agent:CopyPackageAgent \
  --env harbor_ext.local_gpu_docker:LocalGpuDockerEnvironment \
  --agent-timeout 3600 --trial-name "$name" --trials-dir "$EVAL_ROOT/trials-verify"

#!/usr/bin/env bash
# One agent session.
#   run_trial.sh NAME TASK HOURS
# Environment: AGENT_KIND=codex|claude-code; GPUS=task (task.toml, default) or N; GPU_IDS=0,1,...
# (pool, default all); AGENT_CPUS / AGENT_MEMORY_MB override the task's resources; DATA_ENV_FILE
# (default $ENVS_DIR/TASK.env). The separate verifier is deferred: score with run_verify.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
name=$1; task=$2; hours=$3
data_env="${DATA_ENV_FILE:-$ENVS_DIR/$task.env}"
[ -f "$data_env" ] || { echo "No data env file $data_env (run prepare.sh $task DATA_ROOT)" >&2; exit 1; }
mapfile -t agent < <(agent_args); mapfile -t auth < <(agent_env)
resources=()
if [ "${GPUS:-task}" != task ]; then resources+=(--override-gpus "$GPUS"); cap=(-e "HARBOR_GPU_CAP=$GPUS"); else cap=(); fi
[ -n "${AGENT_CPUS:-}" ] && resources+=(--override-cpus "$AGENT_CPUS")
[ -n "${AGENT_MEMORY_MB:-}" ] && resources+=(--override-memory-mb "$AGENT_MEMORY_MB")
mkdir -p "$EVAL_ROOT/trials" "$EVAL_ROOT/research/$name"
$DOCKER run --rm --name "run-$name" $(runner_mounts) -w "$EVAL_ROOT" \
  -e PYTHONPATH="$HARNESS_DIR" -e HARBOR_GPU_DEVICE_IDS="${GPU_IDS:-}" "${cap[@]}" -e HARBOR_SKIP_VERIFIER=1 \
  "${auth[@]}" -e TASK_RESEARCH_DIR="$EVAL_ROOT/research/$name" --env-file "$data_env" \
  "$RUNNER_IMAGE" harbor trial start -p "$TASKS_DIR/$task" "${agent[@]}" \
  --env harbor_ext.local_gpu_docker:LocalGpuDockerEnvironment "${resources[@]}" \
  --agent-timeout "$(python3 -c "print(int(float('$hours') * 3600))")" \
  --trial-name "$name" --trials-dir "$EVAL_ROOT/trials"

#!/usr/bin/env bash
# One command: an agent session on a task, then an independent score of what it published.
#   run.sh TASK [NAME]
# Environment: AGENT_KIND=codex|claude-code (default codex), HOURS (default 24), GPUS=task|N,
# GPU_IDS, plus everything in config.sh. Results land in $EVAL_ROOT/results/NAME/:
#   reward.json, report.json   the verifier's scores
#   submission/                the package the agent published
#   agent/                     the agent's trajectory and session logs
#   notes/                     the agent's markdown notes from /workspace/research
#   snapshots/                 every version the agent published, by seconds since start
# Score snapshots afterwards for reward over time: score_snapshots.sh NAME [K].
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/config.sh"
task=$1; name=${2:-$task-$AGENT_KIND-$(date -u +%Y%m%d-%H%M%S)}
out="$EVAL_ROOT/results/$name"; mkdir -p "$out"; echo "$task" > "$out/task"; echo "$TASKS_DIR" > "$out/tasks_dir"
echo "[run] $name: $AGENT_KIND on $task for ${HOURS:-24} h (GPUs: ${GPUS:-task})"
"$here/run_trial.sh" "$name" "$task" "${HOURS:-24}" > "$out/trial.log" 2>&1 &
trial_pid=$!
until $DOCKER ps --format '{{.Names}}' | grep -q "^run-$name$" || ! kill -0 $trial_pid 2>/dev/null; do sleep 5; done
"$here/snapshot.sh" "$name" "$out" > "$out/snapshot.log" 2>&1 &
wait $trial_pid || true
wait || true
echo "[run] $(($(wc -l < "$out/snapshots/index.tsv" 2>/dev/null || echo 1) - 1)) snapshot(s) kept"
trial="$EVAL_ROOT/trials/$name"
$DOCKER run --rm -v "$EVAL_ROOT:$EVAL_ROOT" --entrypoint bash "$RUNNER_IMAGE" -c "
  set -e; cp -a '$trial/agent' '$out/agent' 2>/dev/null || true
  if [ -d '$trial/artifacts/workspace/submission' ]; then cp -a '$trial/artifacts/workspace/submission' '$out/submission'; fi
  mkdir -p '$out/notes'; cd '$EVAL_ROOT/research/$name' 2>/dev/null && find . -name '*.md' -size -2M -exec cp --parents {} '$out/notes/' \; || true
  chown -R $(id -u):$(id -g) '$out'"
if [ ! -d "$out/submission" ] || [ -z "$(ls -A "$out/submission")" ]; then
  echo "[run] no package was published; see $out/trial.log"; exit 1
fi
echo "[run] scoring the published package"
"$here/run_verify.sh" "verify-$name" "$task" "$out/submission" > "$out/verify.log" 2>&1 || true
$DOCKER run --rm -v "$EVAL_ROOT:$EVAL_ROOT" --entrypoint bash "$RUNNER_IMAGE" -c "
  for f in reward.json report.json; do cp '$EVAL_ROOT/trials-verify/verify-$name/verifier/'\$f '$out/' 2>/dev/null || true; done
  chown -R $(id -u):$(id -g) '$out'"
echo "[run] results in $out"; cat "$out/reward.json" 2>/dev/null || echo "[run] no reward; see $out/verify.log"

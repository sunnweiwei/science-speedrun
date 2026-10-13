#!/usr/bin/env bash
# Install check: run the self-test task end to end (agent session with the resume loop, then the
# separate verifier on the published package). Expect reward 1 and a few resumed turns.
#   AGENT_KIND=codex|claude-code selftest.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/config.sh"
mkdir -p "$ENVS_DIR"; printf 'X=1\n' > "$ENVS_DIR/resume-smoke.env"
TASKS_DIR="$here/selftest" HOURS=0.15 GPUS=0 SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-10} "$here/run.sh" resume-smoke "selftest-$AGENT_KIND-$(date -u +%H%M%S)"

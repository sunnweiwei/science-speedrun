#!/usr/bin/env bash
set -euo pipefail
test ! -e /logs/verifier/reward.json
export HYDROGYM_POLICY_ISOLATION=1
bash /opt/hydrogym-task/evaluation/native_runtime.sh maia python \
    /opt/hydrogym-task/evaluation/harbor_verify.py \
    --data /data --submission /workspace/submission --output /logs/verifier/replay
cp /logs/verifier/replay/reward.json /logs/verifier/reward.json

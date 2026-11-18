#!/usr/bin/env bash
set -euo pipefail
(cd /tests/sealed && sha256sum --quiet -c SHA256SUMS) || { echo "sealed verifier files are missing or differ from SHA256SUMS; see tests/sealed/README.md" >&2; exit 1; }
test ! -e /logs/verifier/reward.json
export HYDROGYM_POLICY_ISOLATION=1
export HYDROGYM_BLIND_SEEDS_FILE="${HYDROGYM_BLIND_SEEDS_FILE:-/tests/sealed/blind_flow_seeds.json}"
bash /opt/hydrogym-task/evaluation/native_runtime.sh maia python \
    /opt/hydrogym-task/evaluation/harbor_verify.py \
    --data /data --submission /workspace/submission --output /logs/verifier/replay
cp /logs/verifier/replay/reward.json /logs/verifier/reward.json
if [ -f /logs/verifier/replay/report.json ]; then
    cp /logs/verifier/replay/report.json /logs/verifier/report.json
fi

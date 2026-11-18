#!/usr/bin/env bash
# Trusted Harbor verifier entry: full suite, sealed perturbations, nonlinear solver.
set -euo pipefail
(cd /tests/sealed && sha256sum --quiet -c SHA256SUMS) || { echo "sealed verifier files are missing or differ from SHA256SUMS; see tests/sealed/README.md" >&2; exit 1; }
test ! -e /logs/verifier/reward.json
python /opt/tokamak-task/evaluation/harbor_verify.py \
    --submission /workspace/submission \
    --output /logs/verifier/replay \
    --randomization /tests/sealed/perturbations.json
cp /logs/verifier/replay/reward.json /logs/verifier/reward.json
if [ -f /logs/verifier/replay/report.json ]; then
    cp /logs/verifier/replay/report.json /logs/verifier/report.json
fi

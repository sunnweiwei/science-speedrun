#!/usr/bin/env bash
# Trusted Harbor verifier entry: full suite, sealed perturbations, nonlinear solver.
set -euo pipefail
test ! -e /logs/verifier/reward.json
python /opt/tokamak-task/evaluation/harbor_verify.py \
    --submission /workspace/submission \
    --output /logs/verifier/replay \
    --randomization /tests/sealed/perturbations.json \
    --workers "${TOKAGYM_WORKERS:-35}"
cp /logs/verifier/replay/reward.json /logs/verifier/reward.json

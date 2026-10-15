#!/usr/bin/env bash
# Minimal oracle: package the trivial fixed-statistic predictor and publish it
# atomically to /workspace/submission — the same "publish a usable candidate
# early" step instruction.md asks an agent to take first. Expected held-out
# reward: 0.2874 (the trivial control). Useful as a Harbor smoke run:
#   harbor run --path <task> --agent oracle
set -euxo pipefail
stage=$(mktemp -d)
cp /workspace/baseline/trivial/infer.sh /workspace/baseline/trivial/predict.py "$stage/"
cp /workspace/data/stats.json "$stage/stats.json"
cat > "$stage/README.md" <<'DONE'
Trivial fixed-statistic control predictor, published by the oracle solution.
No learned parameters; per-task constants from /workspace/data/stats.json. Expected
held-out reward 0.2874. This exists so a valid submission always exists.
DONE
chmod 755 "$stage/infer.sh"
python3 /workspace/baseline/starter/publish.py "$stage" --to /workspace/submission
# prove the published package actually runs on development inputs
out=$(mktemp -d)
bash /workspace/submission/infer.sh /workspace/data/tune "$out"
test -d "$out/predictions"

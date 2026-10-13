#!/usr/bin/env bash
# 1 if the package holds a non-empty answer.txt, else 0.
mkdir -p /logs/verifier
if [ -s /workspace/submission/answer.txt ]; then r=1; else r=0; fi
echo "{\"reward\": $r}" > /logs/verifier/reward.json

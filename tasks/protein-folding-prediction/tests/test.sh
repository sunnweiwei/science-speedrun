#!/usr/bin/env bash
set -euo pipefail
exec python /tests/grade.py \
    --submission /workspace/submission \
    --suite /benchmark/suite \
    --features /benchmark/features.esmc6b_final_int8.jsonl \
    --ground-truth /benchmark/ground_truths \
    --output /logs/verifier

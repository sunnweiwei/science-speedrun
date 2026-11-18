#!/usr/bin/env bash
set -euo pipefail
exec python /tests/grade.py \
    --phase infer \
    --submission /workspace/submission \
    --suite /benchmark/suite \
    --features /benchmark/features.esmc6b_final_int8.jsonl \
    --sealed /sealed/run \
    --results /results \
    --output /logs/verifier

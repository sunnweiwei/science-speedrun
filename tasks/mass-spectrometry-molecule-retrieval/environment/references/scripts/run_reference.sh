#!/usr/bin/env bash
# Run the two-stage reference on the task's unlabelled verifier input.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 INPUT_ROOT OUTPUT_ROOT ASSET_ROOT" >&2
  echo "ASSET_ROOT must contain mist-cf/, FRIGID/, sirius/, and checkpoints/." >&2
  exit 2
fi

input_root="$1"
output_root="$2"
asset_root="$3"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
reference_python="${MASS_SPEC_REFERENCE_PYTHON:-python}"
device="${MASS_SPEC_REFERENCE_DEVICE:-cuda}"
formula_top_k="${MASS_SPEC_FORMULA_TOP_K:-5}"
formula_aggregation="${MASS_SPEC_FORMULA_AGGREGATION:-max}"
fingerprint_threshold="${MASS_SPEC_FINGERPRINT_THRESHOLD:-0.20346222817897797}"
mist_checkpoint="${MASS_SPEC_MIST_CHECKPOINT:-$asset_root/checkpoints/mist_msg.pt}"
scratch_root="$(mktemp -d "${TMPDIR:-/tmp}/massspecgym-reference.XXXXXX")"
trap 'rm -rf -- "$scratch_root"' EXIT

mkdir -p "$output_root"

"$reference_python" "$script_root/predict_mistcf.py" \
  --queries "$input_root/queries.jsonl" \
  --mist-cf-src "$asset_root/mist-cf/src" \
  --checkpoint "$asset_root/checkpoints/mist_cf_msg.ckpt" \
  --fast-filter-checkpoint "$asset_root/checkpoints/mist_cf_fast_filter.ckpt" \
  --sirius "$asset_root/sirius/bin/sirius" \
  --candidate-cache "$scratch_root/formula_candidates.json" \
  --output "$scratch_root/formula_predictions.jsonl" \
  --metrics "$scratch_root/formula_metrics.json" \
  --device "$device" \
  --ppm 10 \
  --max-candidates 256 \
  --top-k 5 \
  --sirius-cores "${MASS_SPEC_SIRIUS_CORES:-16}"

"$reference_python" "$script_root/predict_mist.py" \
  --queries "$input_root/queries.jsonl" \
  --candidate-pools "$input_root/candidate_pools.jsonl" \
  --formula-predictions "$scratch_root/formula_predictions.jsonl" \
  --mist-src "$asset_root/FRIGID/src" \
  --checkpoint "$mist_checkpoint" \
  --output "$output_root/rankings.jsonl" \
  --metrics "$scratch_root/retrieval_metrics.json" \
  --device "$device" \
  --batch-size "${MASS_SPEC_MIST_BATCH_SIZE:-256}" \
  --formula-top-k "$formula_top_k" \
  --formula-aggregation "$formula_aggregation" \
  --similarity tanimoto \
  --threshold "$fingerprint_threshold"

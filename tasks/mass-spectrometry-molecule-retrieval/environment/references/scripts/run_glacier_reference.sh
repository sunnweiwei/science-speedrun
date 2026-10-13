#!/usr/bin/env bash
# Run released GLACIER on its published domain and clean MIST elsewhere.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 INPUT_ROOT OUTPUT_ROOT ASSET_ROOT" >&2
  exit 2
fi

input_root="$1"
output_root="$2"
asset_root="$3"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
glacier_python="${MASS_SPEC_GLACIER_PYTHON:-$asset_root/ms-pred/.venv/bin/python}"
if [[ ! -x "$glacier_python" ]]; then
  echo "GLACIER Python is not executable: $glacier_python" >&2
  exit 2
fi
scratch_root="$(mktemp -d "${TMPDIR:-/tmp}/massspecgym-glacier-reference.XXXXXX")"
trap 'rm -rf -- "$scratch_root"' EXIT

if [[ -z "${MASS_SPEC_MIST_CHECKPOINT:-}" ]]; then
  echo "MASS_SPEC_MIST_CHECKPOINT must point to the task-trained clean MIST encoder" >&2
  exit 2
fi
clean_mist_sha256=30715dbd198bb1d0d776a5dcb9d37488b8d5480c243a1572d62ed27b73e92c6b
actual_mist_sha256="$(sha256sum "$MASS_SPEC_MIST_CHECKPOINT" | cut -d' ' -f1)"
if [[ "$actual_mist_sha256" != "$clean_mist_sha256" ]]; then
  echo "clean MIST checkpoint SHA-256 mismatch: $actual_mist_sha256" >&2
  exit 1
fi
glacier_checkpoint="$asset_root/checkpoints/glacier_msg_no_cf.ckpt"
glacier_checkpoint_sha256=5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11
actual_glacier_sha256="$(sha256sum "$glacier_checkpoint" | cut -d' ' -f1)"
if [[ "$actual_glacier_sha256" != "$glacier_checkpoint_sha256" ]]; then
  echo "GLACIER checkpoint SHA-256 mismatch: $actual_glacier_sha256" >&2
  exit 1
fi
export MASS_SPEC_FINGERPRINT_THRESHOLD="${MASS_SPEC_FINGERPRINT_THRESHOLD:-0.22049389779567719}"

# The existing task-owned clean MIST-CF -> MIST implementation covers only the
# complement of GLACIER's released domain, avoiding redundant full-input work.
"$glacier_python" "$script_root/stage_glacier_fallback.py" \
  --input-root "$input_root" \
  --output-root "$scratch_root/fallback_input"
if [[ -s "$scratch_root/fallback_input/queries.jsonl" ]]; then
  MASS_SPEC_REFERENCE_PYTHON="${MASS_SPEC_REFERENCE_PYTHON:-$glacier_python}" \
    bash "$script_root/run_reference.sh" \
    "$scratch_root/fallback_input" "$scratch_root/fallback" "$asset_root"
else
  mkdir "$scratch_root/fallback"
  touch "$scratch_root/fallback/rankings.jsonl"
fi

"$glacier_python" "$script_root/prepare_glacier_candidates.py" \
  --input-root "$input_root" \
  --output "$scratch_root/glacier_candidates.tsv" \
  --eligible-only

if [[ "$(wc -l < "$scratch_root/glacier_candidates.tsv")" -gt 1 ]]; then
  (
    cd "$asset_root/ms-pred"
    "$glacier_python" src/ms_pred/glacier/predict_smis_joint.py \
      --batch-size "${MASS_SPEC_GLACIER_BATCH_SIZE:-16}" \
      --dataset-labels "$scratch_root/glacier_candidates.tsv" \
      --checkpoint "$glacier_checkpoint" \
      --save-dir "$scratch_root/glacier_predictions" \
      --out-name preds.hdf5 \
      --sparse-out \
      --sparse-k 100 \
      --num-cpu-workers "${MASS_SPEC_GLACIER_CPU_WORKERS:-2}" \
      --feat-cache-size "${MASS_SPEC_GLACIER_FEATURE_CACHE_SIZE:-64}" \
      --prefetch-factor "${MASS_SPEC_GLACIER_PREFETCH_FACTOR:-1}" \
      --gpu
  )

  (
    cd "$asset_root/ms-pred"
    "$glacier_python" "$script_root/score_glacier_predictions.py" \
      --input-root "$input_root" \
      --pred-file "$scratch_root/glacier_predictions/preds.hdf5" \
      --output-root "$scratch_root/glacier_rankings" \
      --distance entropy
  )
else
  mkdir "$scratch_root/glacier_rankings"
  touch "$scratch_root/glacier_rankings/rankings.jsonl"
fi

"$glacier_python" "$script_root/merge_glacier_fallback.py" \
  --input-root "$input_root" \
  --glacier "$scratch_root/glacier_rankings/rankings.jsonl" \
  --fallback "$scratch_root/fallback/rankings.jsonl" \
  --output-root "$output_root"

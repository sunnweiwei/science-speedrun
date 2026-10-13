#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PUBLIC_OUTPUT_ROOT PRIVATE_LABEL_ROOT" >&2
  exit 2
fi

massspec_public_root=$1
massspec_private_root=$2
massspec_revision=d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a
massspec_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
massspec_mode=${MASSSPECGYM_MATERIALIZE_MODE:-prepared}
# Operator-only preparation has network access; development and verifier images
# stay offline. Scope these overrides to this helper process.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
if [[ "$massspec_mode" == prepared ]]; then
  exec python3 "$massspec_script_dir/materialize_hf.py" \
    --public-root "$massspec_public_root" --private-label-root "$massspec_private_root"
elif [[ "$massspec_mode" != source ]]; then
  echo "MASSSPECGYM_MATERIALIZE_MODE must be prepared or source" >&2
  exit 2
fi
massspec_parent_dir=$(dirname -- "$massspec_public_root")
massspec_cache_dir=${MASSSPECGYM_CACHE_DIR:-"$massspec_parent_dir/massspecgym-source-$massspec_revision"}

command -v hf >/dev/null 2>&1 || {
  echo "The Hugging Face 'hf' CLI is required." >&2
  exit 1
}

mkdir -p -- "$massspec_cache_dir"
hf download roman-bushuiev/MassSpecGym \
  --repo-type dataset \
  --revision "$massspec_revision" \
  data/MassSpecGym1.5.tsv \
  data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json \
  --local-dir "$massspec_cache_dir"

python3 "$massspec_script_dir/materialize.py" \
  --source-root "$massspec_cache_dir" \
  --output-root "$massspec_public_root" \
  --private-label-root "$massspec_private_root" \
  --reuse-valid-existing

if [[ "${MASSSPECGYM_PREPARE_SYNTHETIC:-1}" == 1 ]]; then
  bash "$massspec_script_dir/prepare_synthetic_augmentation.sh" \
    "$massspec_public_root" \
    "$massspec_cache_dir/data/MassSpecGym1.5.tsv"
else
  echo "synthetic augmentation skipped (core-data smoke mode only)" >&2
fi

# GLACIER MAGMa supervision reads development/ produced above, so it runs last.
if [[ "${MASSSPECGYM_PREPARE_GLACIER:-1}" == 1 ]]; then
  "${MASSSPECGYM_GLACIER_PYTHON:-/opt/glacier/bin/python}" \
    "$massspec_script_dir/../starter/glacier/prepare.py" \
    --input-root "$massspec_public_root/development" \
    --output-root "$massspec_public_root/development/glacier" \
    --source-root "${MASSSPECGYM_GLACIER_SOURCE:-/opt/ms-pred}" \
    --workers "${MASSSPECGYM_PREPARE_WORKERS:-16}"
else
  echo "GLACIER supervision skipped (core-data smoke mode only)" >&2
fi

#!/usr/bin/env bash
# Download the prepared release from Hugging Face instead of rebuilding it from the sources.
#   fetch_release.sh DATA_ROOT [--dev-only]
# Pulls dev/ (and private/, unless --dev-only) at the pinned revision below, then SE-600M at its
# pinned revision into dev/se600m/. Verifies sizes/SHA-256 against dev/manifest.json.
set -euo pipefail
root=${1:?DATA_ROOT}; dev_only=${2:-}
here=$(cd "$(dirname "$0")" && pwd)
py=${VCZ_MATERIALIZE_PYTHON:-python3}
hf=("$py" -m huggingface_hub.cli.hf)   # huggingface_hub >= 1.0
release=$("$py" -c 'import json,sys; m=json.load(open(sys.argv[1]))["files"]; print(m["prepared_release"]["hf_repo"], m["prepared_release"]["hf_revision"])' "$here/download-source.json")
repo=${release% *}; rev=${release#* }
mkdir -p "$root"
include=(--include "dev/*" --include "manifest.json"); [[ "$dev_only" == "--dev-only" ]] || include+=(--include "private/*")
"${hf[@]}" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$root" "${include[@]}"
mkdir -p "$root/dev/se600m"
"${hf[@]}" download arcinstitute/SE-600M --revision 5a9a80f --local-dir "$root/dev/se600m" \
  --include se600m_epoch16.ckpt --include config.yaml --include protein_embeddings.pt \
  --include MODEL_LICENSE.md --include MODEL_ACCEPTABLE_USE_POLICY.md
"$py" "$here/verify_release.py" --root "$root" ${dev_only:+--dev-only}
echo "release ready: $root"

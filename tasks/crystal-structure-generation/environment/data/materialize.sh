#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: materialize.sh OUTPUT_DIR" >&2
  exit 2
fi

OUTPUT_DIR=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CRYSTALGEN_TMP=$(mktemp -d /tmp/crystalgen-mp20.XXXXXX)
trap 'rm -rf -- "$CRYSTALGEN_TMP"' EXIT

git init "$CRYSTALGEN_TMP/cdvae"
git -C "$CRYSTALGEN_TMP/cdvae" remote add origin https://github.com/txie-93/cdvae.git
git -C "$CRYSTALGEN_TMP/cdvae" fetch --depth 1 origin f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea
git -C "$CRYSTALGEN_TMP/cdvae" checkout --detach FETCH_HEAD

python3 "$SCRIPT_DIR/prepare.py" \
  --source-root "$CRYSTALGEN_TMP/cdvae/data/mp_20" \
  --output-root "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/validate.py" "$OUTPUT_DIR"

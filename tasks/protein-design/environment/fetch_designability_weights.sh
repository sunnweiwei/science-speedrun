#!/usr/bin/env bash
# Download every model parameter the designability criterion needs.
#
#   bash fetch_designability_weights.sh [target-dir]      # default: /assets
#
# Designability = LigandMPNN inverse folding + RoseTTAFold3 refolding. Both sets of
# parameters are public downloads from files.ipd.uw.edu, so this script is all an
# operator needs on ANY cluster - there is nothing site-specific here, and nothing that
# has to be received under a separate agreement.
#
# It is run once at image build time, and can be re-run anywhere: it is idempotent,
# verifies sha256 for every file, and re-downloads only what is missing or corrupt.
#
# Layout it produces (what `tests/pipeline.py::verify_assets` expects):
#
#   <target>/ligandmpnn_params/proteinmpnn_v_48_020.pt
#   <target>/ligandmpnn_params/ligandmpnn_v_32_010_25.pt
#   <target>/rf3/rf3_foundry_01_24_latest_remapped.ckpt
#
# The matching SOURCE is fetched separately: LigandMPNN is cloned at a pinned commit by
# the Dockerfile, and RF3 is vendored in `environment/evaluation/rf3`.
set -euo pipefail

TARGET=${1:-/assets}
MPNN_DIR="$TARGET/ligandmpnn_params"
RF3_DIR="$TARGET/rf3"

# filename <TAB> sha256 <TAB> url
ASSETS=$(cat <<'EOF'
proteinmpnn_v_48_020.pt	c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd	https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt	MPNN
ligandmpnn_v_32_010_25.pt	161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970	https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt	MPNN
rf3_foundry_01_24_latest_remapped.ckpt	364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0	https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt	RF3
EOF
)

mkdir -p "$MPNN_DIR" "$RF3_DIR"

fetch() {
    local name=$1 want=$2 url=$3 dir=$4
    local path="$dir/$name"
    if [ -f "$path" ]; then
        local have
        have=$(sha256sum "$path" | cut -d' ' -f1)
        if [ "$have" = "$want" ]; then
            echo "  ok      $name (already present, sha256 verified)"
            return 0
        fi
        echo "  REDO    $name (sha256 mismatch, re-downloading)"
        rm -f "$path"
    fi
    echo "  fetch   $name <- $url"
    # --fail so an HTML error page is never mistaken for a checkpoint.
    curl -sS --fail --retry 3 --retry-delay 5 -o "$path.part" "$url"
    local got
    got=$(sha256sum "$path.part" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        rm -f "$path.part"
        echo "  FAIL    $name sha256 mismatch: expected $want, got $got" >&2
        return 1
    fi
    mv "$path.part" "$path"
    echo "  ok      $name ($(stat -c%s "$path") bytes, sha256 verified)"
}

echo "Fetching designability weights into $TARGET"
while IFS=$'\t' read -r name sha url kind; do
    [ -n "$name" ] || continue
    case "$kind" in
        MPNN) fetch "$name" "$sha" "$url" "$MPNN_DIR" ;;
        RF3)  fetch "$name" "$sha" "$url" "$RF3_DIR" ;;
    esac
done <<< "$ASSETS"

echo
echo "done. Point the grader at it with:"
echo "  python tests/grader.py ... --assets $TARGET"
echo "or, for the starter's self-scoring path, export:"
echo "  PROTEIN_DESIGN_ASSETS=$TARGET"

#!/usr/bin/env bash
# Contract entry point: bash infer.sh INPUT_ROOT OUTPUT_ROOT
#
# Finds the checkpoint that ships beside this script and hands the real work to
# rfd3_infer.sh, which drives upstream RFD3 unmodified. The contract gives no slot for a
# checkpoint argument, so the location is a convention, resolved in this order:
#
#   1. $RFD3_CKPT, if set
#   2. <this dir>/ckpt/*.ckpt        - newest by mtime
#   3. <this dir>/*.ckpt             - newest by mtime
#
# Train into the package so that resolution succeeds: instruction.md requires the
# submission to carry one current candidate, and the verifier runs this script from an
# unspecified working directory, so nothing outside this directory is reachable.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: infer.sh INPUT_ROOT OUTPUT_ROOT" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the vendored rfd3/foundry source from THIS directory, not from the image.
#
# The verifier launches this script with an explicit environment
# (`subprocess.Popen(env=...)` in tests/main_verifier.py), which REPLACES the
# environment rather than extending it - so the Dockerfile's `ENV PYTHONPATH` and
# `PROJECT_ROOT` do not arrive here. The image installs rfd3's dependencies by name but
# deliberately does not `pip install rc-foundry`, so `import rfd3` resolves through
# PYTHONPATH alone and fails without this. The contract asks for a relocatable package
# that resolves package-owned files relative to itself; this is that.
#
# Package-owned paths go FIRST, so a relocated copy uses its own vendored source rather
# than whatever the host happens to have installed - which is the point of vendoring.
export PROJECT_ROOT="$here/rfd3"
export PYTHONPATH="$here/rfd3/src:$here/rfd3/models/rfd3/src${PYTHONPATH:+:$PYTHONPATH}"

pick_ckpt() {
    if [[ -n "${RFD3_CKPT:-}" ]]; then
        printf '%s\n' "$RFD3_CKPT"
        return
    fi
    local found
    found=$(ls -t "$here"/ckpt/*.ckpt "$here"/*.ckpt 2>/dev/null | head -n 1 || true)
    printf '%s\n' "$found"
}

ckpt=$(pick_ckpt)
if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "infer.sh: no checkpoint found." >&2
    echo "  looked at: \$RFD3_CKPT, $here/ckpt/*.ckpt, $here/*.ckpt" >&2
    echo "  the task ships none - train one first (see README.md) and copy or symlink" >&2
    echo "  it to $here/ckpt/, or set RFD3_CKPT." >&2
    exit 3
fi

echo "infer.sh: using checkpoint $ckpt" >&2

# rfd3_infer.sh builds a whole work tree (inputs/, designs/, submission/) and the
# contract wants only `predictions/` at OUTPUT_ROOT, so stage elsewhere and lift that
# one subtree up. rfd3_collect.py writes submission/predictions/{<track>.jsonl,
# structures/*.pdb}, which is the contract layout exactly.
staging=$(mktemp -d "${TMPDIR:-/tmp}/infer.XXXXXX")
trap 'rm -rf "$staging"' EXIT

bash "$here/rfd3_infer.sh" "$ckpt" "$1" "$staging"

mkdir -p "$2"
cp -a "$staging/submission/predictions" "$2/"

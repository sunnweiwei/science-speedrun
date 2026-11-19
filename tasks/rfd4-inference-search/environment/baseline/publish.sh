#!/usr/bin/env bash
set -euo pipefail
src=${1:?Usage: publish.sh BASELINE_DIRECTORY [DESTINATION]}
out=${2:-/workspace/submission}
mkdir -p "$out"
tmp=$(mktemp "$out/.policy.XXXXXX")
trap 'rm -f "$tmp"' EXIT
cp "$src/policy.json" "$tmp"
chmod 644 "$tmp"
mv -fT "$tmp" "$out/policy.json"

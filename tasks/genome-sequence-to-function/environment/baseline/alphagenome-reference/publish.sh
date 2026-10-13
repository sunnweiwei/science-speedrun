#!/usr/bin/env bash
# publish.sh PACKAGE [TARGET=/workspace/submission]
# Copy PACKAGE next to TARGET, check that it is complete, then swap it in with one rename so the
# controller never snapshots a half-written submission.
set -euo pipefail
PKG=$1; TARGET=${2:-/workspace/submission}
parent=$(dirname "$TARGET"); staged=$(mktemp -d "$parent/.submission.XXXXXX")
cp -a "$PKG"/. "$staged"/
for f in infer.sh perturb.sh README.md; do [[ -s "$staged/$f" ]] || { echo "missing $f" >&2; exit 2; }; done
[[ -x "$staged/infer.sh" && -x "$staged/perturb.sh" ]] || { echo "entry points not executable" >&2; exit 2; }
if [[ -d "$TARGET" ]]; then
  old="$parent/.submission.old.$$"
  # A directory from a lower image layer cannot be renamed on overlayfs; empty it instead.
  if mv "$TARGET" "$old" 2>/dev/null; then mv "$staged" "$TARGET"; rm -rf "$old"
  else find "$TARGET" -mindepth 1 -delete; cp -a "$staged"/. "$TARGET"/; rm -rf "$staged"; fi
else
  mv "$staged" "$TARGET"
fi
echo "published $PKG -> $TARGET"

#!/usr/bin/env bash
# Watch a running session's /workspace/submission and keep a copy of every published version.
#   snapshot.sh NAME OUT_DIR   (run.sh starts it in the background)
# Every SNAPSHOT_INTERVAL seconds (default 60) it hashes the package's file list (path, size, mtime);
# on a change it copies the package, hashes again, and keeps the copy only if nothing moved during
# the copy. Snapshots land in OUT_DIR/snapshots/<seconds since start>/ with an index.tsv.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
name=$1; out=$2/snapshots; interval=${SNAPSHOT_INTERVAL:-60}
mkdir -p "$out"; [ -f "$out/index.tsv" ] || printf 'seconds\tutc\tdigest\tbytes\n' > "$out/index.tsv"
ctr="${name}__env-main-1"; start=$(date +%s); last=""
digest() { $DOCKER exec "$ctr" sh -c 'cd /workspace/submission 2>/dev/null && find . -printf "%P %s %T@ %l\n" | sort | sha256sum' 2>/dev/null | cut -c1-16; }
while $DOCKER ps --format '{{.Names}}' | grep -q "^run-$name$"; do
  if $DOCKER ps --format '{{.Names}}' | grep -q "^$ctr$"; then
    d=$(digest)
    if [ -n "$d" ] && [ "$d" != "$last" ] && [ "$d" != "e3b0c44298fc1c14" ] && \
       [ -n "$($DOCKER exec "$ctr" sh -c 'ls -A /workspace/submission' 2>/dev/null)" ]; then
      s=$(( $(date +%s) - start )); tmp="$out/.tmp-$s"; mkdir -p "$tmp"
      if $DOCKER exec "$ctr" tar -C /workspace/submission -cf - . 2>/dev/null | tar -C "$tmp" -xf - 2>/dev/null \
         && [ "$(digest)" = "$d" ]; then
        mv "$tmp" "$out/$s"; last=$d
        printf '%s\t%s\t%s\t%s\n' "$s" "$(date -u +%FT%TZ)" "$d" "$(du -sb "$out/$s" | cut -f1)" >> "$out/index.tsv"
      else
        rm -rf "$tmp"   # changed during the copy: try again next tick
      fi
    fi
  fi
  sleep "$interval"
done
